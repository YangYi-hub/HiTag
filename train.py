import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from hitag.models import hitag, hycoclip
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from hitag.data import create_obj_dataset, create_sampler, create_loader

import clip

from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from hitag.models.text_encoders import TransformerTextEncoder
from hitag.models.image_encoders import build_timm_vit
from hitag.models.tokenizer import Tokenizer
from torch.amp import GradScaler, autocast


def custom_collate_fn(batch):
    images, captions, image_tags, tag_trees = zip(*batch)

    collated_images = default_collate(images)
    collated_image_tags = default_collate(image_tags) if image_tags[0] is not None else None

    
    collated_tag_trees = pad_sequence(tag_trees, batch_first=True, padding_value=-1)

    collated_captions = list(captions)
    
    return collated_images, collated_captions, collated_image_tags, collated_tag_trees

def train_hitag(model, data_loader, optimizer, epoch, device, config, model_hycoclip, log_file_path, tokenizer):
    model.train()  
    
    scaler = GradScaler()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('contrastive_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('entailment_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_tag', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_dis', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_alignment', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('text_image_entailment_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tag_image_entailment_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hirechical_tag_entailment_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('cap_tag_entailment_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('_curv', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('entail_weight', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image, caption, image_tag, tag_tree) in enumerate(metric_logger.log_every(data_loader, print_freq, header, log_file_path)):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()

        batch_text_tokens = tokenizer(caption)
        batch_text_embed = model_hycoclip.encode_text(batch_text_tokens, project = False)
        
        image = image.to(device,non_blocking=True)
        
        hycoclip_image_feature = model_hycoclip.encode_image(image, project = False)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, contrastive_loss, loss_tag, loss_alignment, loss_dis, text_image_entailment_loss, tag_image_entailment_loss, hirechical_tag_entailment_loss, cap_tag_entailment_loss, entailment_loss, _curv, entail_weight = \
                model(image, caption, image_tag, batch_text_embed, tag_tree, hycoclip_image_feature)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()   

        metric_logger.update(loss=loss.item())
        metric_logger.update(contrastive_loss=contrastive_loss.item())
        metric_logger.update(text_image_entailment_loss=text_image_entailment_loss.item())
        metric_logger.update(tag_image_entailment_loss=tag_image_entailment_loss.item())
        metric_logger.update(hirechical_tag_entailment_loss=hirechical_tag_entailment_loss.item())
        metric_logger.update(cap_tag_entailment_loss=cap_tag_entailment_loss.item())
        metric_logger.update(entailment_loss=entailment_loss.item())
        metric_logger.update(loss_dis=loss_dis.item())
        metric_logger.update(_curv=_curv.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_tag=loss_tag.item())
        metric_logger.update(loss_alignment=loss_alignment.item())
        metric_logger.update(entail_weight=entail_weight)#.item())

        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_obj_dataset('pretrain', config, min_scale=0.2)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)  
    
    config['init_lr'] = float(config['init_lr'])
    config['min_lr'] = float(config['min_lr'])
    config['warmup_lr'] = float(config['warmup_lr']) 
    config['curv_init'] = float(config['curv_init'])
    config['entail_weight'] = float(config['entail_weight']) 

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[custom_collate_fn])[0]    
    log_file_path = os.path.join(args.output_dir, "train_log.txt")
    tokenizer = Tokenizer()
    
    #### Model #### 
    if args.model_type == 'hitag':
        print("Creating pretrained HycoCLIP model")
        model_hycoclip = hycoclip(pretrained='/pretrain_model/hycoclip_vit_b.pth',
                visual=build_timm_vit(
                    arch="vit_base_patch16_224",
                    global_pool="token",
                    use_sincos2d_pos=True,
                ),
                textual=TransformerTextEncoder(
                    arch="L12_W512", vocab_size=49408, context_length=77
                ),
                embed_dim=config['embed_dim'],
                curv_init=config['curv_init'],
                learn_curv=config['learn_curv'],
                entail_weight=config['entail_weight'],
            )
        print("Creating HiTag model")
        model = hitag(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                vit_ckpt_layer=config['vit_ckpt_layer'], stage = 'train_from_scratch',
                                embed_dim = config['embed_dim'], curv_init = config['curv_init'], learn_curv = config['learn_curv'], entail_weight = config['entail_weight'])

    model = model.to(device)   
    

    model_hycoclip = model_hycoclip.to(device)
    for _, param in model_hycoclip.named_parameters():
        param.requires_grad = False

    model.label_embed.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
        
    print("Start training")
    
    
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])

        if args.model_type == 'hitag':
            train_stats = train_hitag(model, data_loader, optimizer, epoch, device, config, model_hycoclip, log_file_path, tokenizer)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='hitag/configs/pretrain.yaml')
    parser.add_argument("--model-type",type=str, default="hitag")
    parser.add_argument('--output-dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)