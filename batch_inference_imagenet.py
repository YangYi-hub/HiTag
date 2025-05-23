from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os
import json

from hitag import get_transform
from hitag.models import hitag
from hitag.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR, build_openset_hycoclip_llm_label_embedding

from hitag.models import lorentz as L

device = "cuda" if torch.cuda.is_available() else "cpu"


class _Dataset(Dataset):
    def __init__(self, imglist, input_size):
        self.imglist = imglist
        self.transform = get_transform(input_size)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        try:
            img = Image.open(self.imglist[index])#+".jpg")
        except (OSError, FileNotFoundError, UnidentifiedImageError):
            img = Image.new('RGB', (10, 10), 0)
            print("Error loading image:", self.imglist[index])
        return self.transform(img)


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-type",
                        type=str,
                        default="hitag",)
    parser.add_argument("--checkpoint",
                        default="path/to/checkpoint",
                        type=str,)
    parser.add_argument("--backbone",
                        type=str,
                        choices=("swin_l", "swin_b"),
                        default="swin_l",
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set",
                        default=False,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                        ))
    # data
    parser.add_argument("--dataset",
                        type=str,
                        default="imagenet_multi_1000")

    parser.add_argument("--input-size",
                        type=int,
                        default=224)
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold",
                       type=float,
                       default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file",
                       type=str,
                       default=None,
                       help=(
                           "Use custom class-wise thresholds by providing a "
                           "text file. Each line is a float-type threshold, "
                           "following the order of the tags in taglist file. "
                           "See `ram/data/ram_tag_list_threshold.txt` as an "
                           "example. Mutually exclusive with `--threshold`. "
                           "If both `--threshold` and `--threshold-file` is "
                           "`None`, will use default threshold setting."
                       ))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    args.model_type = args.model_type.lower()


    if args.backbone is None:
        args.backbone = "swin_l" 

    return args


def load_dataset(
    dataset: str,
    model_type: str,
    input_size: int,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = str(Path(__file__).resolve().parent / "datasets" / dataset)
    img_root = dataset_root + "/imgs"
    if model_type == "hitag":
        tag_file = dataset_root + f"/{dataset}_taglist.txt"
        annot_file = dataset_root + f"/{dataset}_annots.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    with open(annot_file, "r", encoding="utf-8") as f:
        imglist = [img_root + "/" + line.strip().split(",")[0] for line in f]

    loader = DataLoader(
        dataset=_Dataset(imglist,input_size),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    open_tag_des = dataset_root + f"/{dataset}_tag_description.json"
    if os.path.exists(open_tag_des):
        with open(open_tag_des, 'rb') as fo:
            tag_des = json.load(fo)

    else:
        tag_des = None
    info = {
        "taglist": taglist,
        "imglist": imglist,
        "annot_file": annot_file,
        "img_root": img_root,
        "tag_des": tag_des
    }

    return loader, info


def get_class_idxs(
    model_type: str,
    open_set: bool,
    taglist: List[str]
) -> Optional[List[int]]:

    if model_type == "hitag":
        if not open_set:
            model_taglist_file = "hitag/data/tag_list.txt"
            with open(model_taglist_file, "r", encoding="utf-8") as f:
                model_taglist = [line.strip() for line in f]
            return [model_taglist.index(tag) for tag in taglist]
        else:
            return None
    else: 
        return [int(tag) for tag in taglist]


def load_thresholds(
    threshold: Optional[float],
    threshold_file: Optional[str],
    model_type: str,
    open_set: bool,
    class_idxs: List[int],
    num_classes: int,
) -> List[float]:
    if not threshold_file and not threshold:  
        if model_type == "hitag":
            if not open_set:  
                ram_threshold_file = "hitag/data/tag_list_threshold.txt"
                with open(ram_threshold_file, "r", encoding="utf-8") as f:
                    idx2thre = {
                        idx: float(line.strip()) for idx, line in enumerate(f)
                    }
                    return [idx2thre[idx] for idx in class_idxs]
            else:
                return [0.75] * num_classes
        else:
            return [0.68] * num_classes
    elif threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thresholds = [float(line.strip()) for line in f]
        assert len(thresholds) == num_classes
        return thresholds
    else:
        return [threshold] * num_classes


def gen_pred_file(
    imglist: List[str],
    tags: List[List[str]],
    img_root: str,
    pred_file: str
) -> None:
    """Generate text file of tag prediction results."""
    with open(pred_file, "w", encoding="utf-8") as f:
        for image, tag in zip(imglist, tags):
            # should be relative to img_root to match the gt file.
            s = str(Path(image).relative_to(img_root))
            if tag:
                s = s + "," + ",".join(tag)
            f.write(s + "\n")

def load_hitag(
    backbone: str,
    checkpoint: str,
    input_size: int,
    taglist: List[str],
    tag_des: List[str],
    open_set: bool,
    class_idxs: List[int],
) -> Module:
    model = hitag(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference
    if open_set:
        print("Building tag embeddings ...")
        label_embed, _ = build_openset_hycoclip_llm_label_embedding(tag_des)
        model.label_embed = Parameter(label_embed.float())
        model.num_class = len(tag_des)
    else:
        model.label_embed = Parameter(model.label_embed.data.reshape(model.num_class,6,512)[class_idxs, :, :].reshape(len(class_idxs)*6, 512))
        model.num_class = len(class_idxs)
    return model.to(device).eval()


@torch.no_grad()
def forward_hitag(model: Module, imgs: Tensor) -> Tensor:
    model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
    _curv = model.curv.exp()

    tag_label_embed = model.label_embed * model.textual_alpha.exp()
    with torch.autocast(model.device.type, dtype=torch.float32):
        tag_label_embed = L.exp_map0(tag_label_embed, model.curv.exp())
    
    image_embeds = model.image_proj(model.visual_encoder(imgs.to(device)))
    bs = image_embeds.shape[0]
    image_atts = torch.ones(image_embeds.size()[:-1],
                            dtype=torch.long).to(imgs.device)

    ##================= Distillation from CLIP ================##
    image_cls_embeds = image_embeds[:, 0, :]
    
    image_cls_embeds = image_cls_embeds * model.visual_alpha.exp()
    image_embeds = image_embeds * model.visual_alpha.exp()
    with torch.autocast(model.device.type, dtype=torch.float32):
        image_embeds = L.exp_map0(image_embeds, model.curv.exp())
        image_cls_embeds = L.exp_map0(image_cls_embeds, model.curv.exp())

    des_per_class = int(tag_label_embed.shape[0] / model.num_class)
    logits_per_image = L.pairwise_inner(image_cls_embeds, tag_label_embed, _curv)
    logits_per_image = logits_per_image.view(bs, -1,des_per_class)
    
    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed = torch.empty(bs, model.num_class, 512).to(imgs.device).to(imgs.dtype)
    
    for i in range(bs):
        reshaped_value = model.label_embed.view(-1, des_per_class, 512)
        product = weight_normalized[i].unsqueeze(-1) * reshaped_value
        label_embed[i] = product.sum(dim=1)

    label_embed_euclid = L.log_map0(label_embed.to(_curv.device), _curv)
    image_embeds_euclid = L.log_map0(image_embeds, _curv)
    
    label_embed_euclid = torch.nn.functional.relu(model.wordvec_proj(label_embed_euclid))


    tagging_embed,_ = model.tagging_head(
        encoder_embeds=label_embed_euclid,
        encoder_hidden_states=image_embeds_euclid,
        encoder_attention_mask=image_atts.to(_curv.device),
        return_dict=False,
        mode='tagging',
    )

    return sigmoid(model.fc(tagging_embed).squeeze(-1))


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


if __name__ == "__main__":
    args = parse_args()

    # set up output paths
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir + "/" + name for name in
        ("pred.txt", "pr.txt", "ap.txt", "summary.txt", "logits.pth")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
            "model_type", "backbone", "checkpoint", "open_set",
            "dataset", "input_size",
            "threshold", "threshold_file",
            "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    # prepare data
    loader, info = load_dataset(
        dataset=args.dataset,
        model_type=args.model_type,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist, imglist, annot_file, img_root, tag_des = \
        info["taglist"], info["imglist"], info["annot_file"], info["img_root"], info["tag_des"]

    # get class idxs
    class_idxs = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist
    )

    # set up threshold(s)
    thresholds = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(taglist)
    )

    # inference
    if Path(logit_file).is_file():

        logits = torch.load(logit_file)#torch.Size([57224, 214])

    else:
        # load model
        
        if args.model_type == "hitag":
            model = load_hitag(
                backbone=args.backbone,
                checkpoint=args.checkpoint,
                input_size=args.input_size,
                taglist=taglist,
                tag_des = tag_des,
                open_set=args.open_set,
                class_idxs=class_idxs
            )
        # inference
        logits = torch.empty(len(imglist), len(taglist))
        pos = 0
        for imgs in tqdm(loader, desc="inference"):
            if args.model_type == "hitag":
                out = forward_hitag(model, imgs)

            bs = imgs.shape[0]
            logits[pos:pos+bs, :] = out.cpu()
            pos += bs

        torch.save(logits, logit_file)

    pred_tags = []
    for scores in logits.tolist():
        pred_tags.append([
            taglist[i] for i, s in enumerate(scores) if s >= thresholds[i]
        ])

    gen_pred_file(imglist, pred_tags, img_root, pred_file)

    mAP, APs = get_mAP(logits.numpy(), annot_file, taglist)
    CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    with open(ap_file, "w", encoding="utf-8") as f:
        f.write("Tag,AP\n")
        for tag, AP in zip(taglist, APs):
            f.write(f"{tag},{AP*100.0:.2f}\n")

    with open(pr_file, "w", encoding="utf-8") as f:
        f.write("Tag,Precision,Recall\n")
        for tag, P, R in zip(taglist, Ps, Rs):
            f.write(f"{tag},{P*100.0:.2f},{R*100.0:.2f}\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, f"mAP: {mAP*100.0}")
        print_write(f, f"CP: {CP*100.0}")
        print_write(f, f"CR: {CR*100.0}")
