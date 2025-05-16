import json
import warnings

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from .bert import BertConfig, BertLMHeadModel, BertModel
from .swin_transformer import SwinTransformer
from .utils import *

from . import lorentz as L

warnings.filterwarnings("ignore")



class HiTag(nn.Module):
    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=384,#224
                 text_encoder_type='bert-base-uncased',
                 vit='base',#swin_l
                 vit_grad_ckpt=False,#
                 vit_ckpt_layer=0,#
                 threshold=0.68,#
                 delete_tag_index=[],#
                 tag_list=f'{CONFIG_PATH}/data/tag_list.txt',
                 stage='eval',
                 embed_dim = 512,
                 curv_init = 1.0,
                 learn_curv = True,
                 entail_weight = 10.0):
        super().__init__()

        if vit == 'swin_b':
            if image_size == 224:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

            if stage == 'train_from_scratch':
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        elif vit == 'swin_l':
            if image_size == 224:##
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

            if stage == 'train_from_scratch':
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        self.tokenizer = init_tokenizer(text_encoder_type)

        self.delete_tag_index = delete_tag_index#[]

        self.tag_list = self.load_tag_list(tag_list)

        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/configs/q2l_config.json')
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))

        if stage == 'train_from_scratch':
            self.label_embed = nn.Parameter(torch.load(f'{CONFIG_PATH}/data/frozen_tag_embedding/tag_des_embedding.pth',map_location='cpu').float())
        else:
            self.label_embed = nn.Parameter(torch.zeros(self.num_class * 6, q2l_config.encoder_width))

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.image_proj = nn.Linear(vision_width, 512)

        self.class_threshold = torch.ones(self.num_class) * self.threshold
        ram_class_threshold_path = f'{CONFIG_PATH}/data/tag_list_threshold.txt'
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value
        self.reweight_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        self.curv = nn.Parameter(
            torch.tensor(curv_init), requires_grad=learn_curv
        )

        self._curv_minmax = {
            "max": math.log(1.0 * 10),
            "min": math.log(1.0 / 10),
        }
        self.entail_weight = entail_weight

        self.visual_alpha = nn.Parameter(torch.tensor(-0.3909))
        self.textual_alpha = nn.Parameter(torch.tensor(-0.3898))

        self.tagging_loss_function = AsymmetricLoss(gamma_neg=7,
                                                    gamma_pos=0,
                                                    clip=0.05)

        self.text_alignment_loss_function = AsymmetricLoss(gamma_neg=4,
                                                    gamma_pos=0,
                                                    clip=0.05)

        self._global_aperture_thresh = 0.7
        self._local_aperture_thresh = 1.2 

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def forward(self, image, caption, image_tag, batch_text_embed, tag_tree, hireclip_feature):

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        batch_text_embed = batch_text_embed * self.textual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            batch_text_embed = L.exp_map0(batch_text_embed, self.curv.exp())

        tag_label_embed = self.label_embed * self.textual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            tag_label_embed = L.exp_map0(tag_label_embed, self.curv.exp())
        
        image_embeds = self.image_proj(self.visual_encoder(image))
        bs = image_embeds.shape[0]
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        loss_dis = F.l1_loss(image_cls_embeds, hireclip_feature)
        
        image_cls_embeds = image_cls_embeds * self.visual_alpha.exp()
        image_embeds = image_embeds * self.visual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            image_embeds = L.exp_map0(image_embeds, self.curv.exp())
            image_cls_embeds = L.exp_map0(image_cls_embeds, self.curv.exp())

            des_per_class = int(tag_label_embed.shape[0] / self.num_class)
            logits_per_image = L.pairwise_inner(image_cls_embeds, tag_label_embed, _curv)
            logits_per_image = logits_per_image.view(bs, -1,des_per_class)
            
            weight_normalized = F.softmax(logits_per_image, dim=2)
            label_embed = torch.empty(bs, self.num_class, 512).to(image.device).to(image.dtype)
            
            for i in range(bs):
                reshaped_value = self.label_embed.view(-1, des_per_class, 512)
                product = weight_normalized[i].unsqueeze(-1) * reshaped_value
                label_embed[i] = product.sum(dim=1)

            label_embed_euclid = L.log_map0(label_embed, _curv)
            batch_text_embed_euclid = L.log_map0(batch_text_embed, _curv)
            image_embeds_euclid = L.log_map0(image_embeds, _curv)
    
    
        label_embed_euclid = torch.nn.functional.relu(self.wordvec_proj(label_embed_euclid))


        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed_euclid,
            encoder_hidden_states=image_embeds_euclid,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        loss_tag = self.tagging_loss_function(logits, image_tag)

        batch_text_embed_euclid = torch.nn.functional.relu(self.wordvec_proj(batch_text_embed_euclid.to(self.label_embed.dtype)))
        batch_text_embed_euclid = batch_text_embed_euclid.unsqueeze(0).repeat(bs, 1, 1)
        alignment_embedding = self.tagging_head(
            encoder_embeds=batch_text_embed_euclid,
            encoder_hidden_states=image_embeds_euclid,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )
        alignment_logits = self.fc(alignment_embedding[0]).squeeze(-1)

        with torch.no_grad():
            alignment_targets = torch.zeros(alignment_logits.size()).to(image.device)
            alignment_targets.fill_diagonal_(1)

        loss_alignment = self.text_alignment_loss_function(alignment_logits,alignment_targets)


        contrastive_loss = loss_tag + loss_alignment + loss_dis
        
        B, C = image_cls_embeds.shape
        device = image_cls_embeds.device

        mask_pos = image_tag.to(torch.bool)              
        tag_embed_all = label_embed[mask_pos]
        batch_idx  = mask_pos.nonzero(as_tuple=True)[0]    
        image_embed_all = image_cls_embeds[batch_idx]
        batch_text_embed_all = batch_text_embed[batch_idx]

        tree = torch.tensor(tag_tree, device=device, dtype=torch.long)
        parents = tree[..., 0]                         
        children= tree[..., 1]                              
        mask_edge = (parents >= 0) & (children >= 0)    

        batch_idx_edge = (
            torch.arange(B, device=device)
                .unsqueeze(1)
                .expand(B, parents.size(1))
        )[mask_edge]                                 

        parent_idx = parents[mask_edge]                 
        child_idx = children[mask_edge]             

        # 从 label_embed 中一次性取出所有 parent / child 嵌入
        parent_tag = label_embed[batch_idx_edge, parent_idx] 
        child_tag  = label_embed[batch_idx_edge,  child_idx]
        
 
        with torch.autocast(self.device.type, dtype=torch.float32):  
            _angle = L.oxy_angle(batch_text_embed, image_cls_embeds, _curv)
            _aperture = L.half_aperture(batch_text_embed, _curv)

            _tag_angle = L.oxy_angle(tag_embed_all, image_embed_all, _curv)
            _tag_aperture = L.half_aperture(tag_embed_all, _curv)
            
            _pc_angle = L.oxy_angle(parent_tag, child_tag, _curv)
            _parent_aperture = L.half_aperture(parent_tag, _curv)
            
            _ct_angle = L.oxy_angle(tag_embed_all, batch_text_embed_all, _curv)
            
            
            text_image_entailment_loss = torch.clamp(_angle - self._global_aperture_thresh * _aperture, min=0).mean()
            tag_image_entailment_loss = torch.clamp(_tag_angle - self._global_aperture_thresh * _tag_aperture, min=0).mean()
            hirechical_tag_entailment_loss = torch.clamp(_pc_angle - self._local_aperture_thresh * _parent_aperture, min=0).mean()
            cap_tag_entailment_loss = torch.clamp(_ct_angle - self._local_aperture_thresh * _tag_aperture, min=0).mean()
        
            entailment_loss = text_image_entailment_loss + tag_image_entailment_loss + hirechical_tag_entailment_loss + cap_tag_entailment_loss

        entailment_loss = self.entail_weight * entailment_loss
        loss = contrastive_loss + entailment_loss

        
        return loss, contrastive_loss, loss_tag, loss_alignment, loss_dis, text_image_entailment_loss, tag_image_entailment_loss, hirechical_tag_entailment_loss, cap_tag_entailment_loss, entailment_loss, _curv, self.entail_weight


    def generate_tag(self,
                 image
                 ):

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.label_embed.shape[0] / self.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = self.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds @ self.label_embed.t())
        logits_per_image = logits_per_image.view(bs, -1,des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.num_class, 512).to(image.device).to(image.dtype)

        for i in range(bs):
            reshaped_value = self.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = torch.nn.functional.relu(self.wordvec_proj(label_embed_reweight))

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []

        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))

        tag2logits = {}
        sig_logits = torch.sigmoid(logits)
        tag_list = self.tag_list
        sig_logits = list(sig_logits.cpu().numpy())
        for n,l in zip(tag_list, sig_logits[0]):
            tag2logits[str(n)] = float(l)


        return tag_output,tag2logits


def hitag(pretrained='', **kwargs):
    model = HiTag(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        elif kwargs['vit'] == 'swin_l':
            model, msg = load_checkpoint_swinlarge(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        print('vit:', kwargs['vit'])
    return model
