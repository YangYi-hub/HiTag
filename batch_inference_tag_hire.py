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
from torch import nn

from hitag import get_transform
from hitag.models import hitag
from hitag.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR, build_openset_hycoclip_llm_label_embedding

from hitag.models import lorentz as L

import networkx as nx
from networkx.algorithms.tree import tree_edit_distance

device = "cuda" if torch.cuda.is_available() else "cpu"


class _Dataset(Dataset):
    def __init__(self, imglist, input_size):
        self.imglist = imglist
        self.transform = get_transform(input_size)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        try:
            img = Image.open(self.imglist[index]+".jpg")
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
                        default="/path/to/checkpoint",
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
                        default="openimages_common_214")

    parser.add_argument("--input-size",
                        type=int,
                        default=224)#384)
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
    parser.add_argument("--batch-size", type=int, default=8)#128)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    args.model_type = args.model_type.lower()

    assert not (args.model_type == "tag2text" and args.open_set)

    if args.backbone is None:
        args.backbone = "swin_l" if args.model_type == "hitag" else "swin_b"

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
        annot_file = dataset_root + f"/{dataset}_tree_annots.json"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    with open(annot_file, "r", encoding="utf-8") as f:
        tag_anno = json.load(f)
        imglist = tag_anno.keys()
        anno_list = tag_anno.values()


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
        "annot_file": anno_list,
        "img_root": img_root,
        "tag_des": tag_des
    }

    return loader, info


def get_class_idxs(
    model_type: str,
    open_set: bool,
    taglist: List[str]
) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
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
    """Decide what threshold(s) to use."""
    if not threshold_file and not threshold:  
        if model_type == "hitag":
            if not open_set: 
                ram_threshold_file = "ram/data/tag_list_threshold.txt"
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

def find_category_hierarchy(category_name, tree, level=0):

    if tree['synset'] == category_name:

        return [{'level': level, 'category': category_name}]

    for child in tree.get('children', []):
        result = find_category_hierarchy(category_name, child, level + 1)
        if result:
            return [{'level': level, 'category': tree['synset']}] + result

    return None

def build_category_paths(category_list, cate_tree):

    paths = []
    for category_name in category_list:
        path = find_category_hierarchy(category_name, cate_tree)
        if path:
            paths.append(path)
    return paths

def merge_paths_to_tree(paths, name_to_index):

    root = {
        "cate_index": -1,
        "level": -1,
        "children": []
    }
    
    for path in paths:
        add_path_to_tree(root, path, name_to_index)
    return simplify_root(root)


def add_path_to_tree(current_node, path, name_to_index):

    if not path:
        return
    
    node_info = path[0]
    node_category = node_info["category"]
    node_level = node_info["level"]
    
    cate_index = name_to_index.get(node_category, -1)

    for child in current_node["children"]:
        if child["cate_index"] == cate_index and child["level"] == node_level:
            add_path_to_tree(child, path[1:], name_to_index)
            return

    new_node = {
        "cate_index": cate_index,
        "level": node_level,
        "children": []
    }
    current_node["children"].append(new_node)
    
    add_path_to_tree(new_node, path[1:], name_to_index)

def simplify_root(root):
    if root["level"] == -1:
        if len(root["children"]) == 1:
            return root["children"][0]  
        else:

            return {
                "cate_index": -1,  
                "level": 0,
                "children": root["children"]
            }
    else:
        return root

def collect_all_indices_from_tree(tree):
 
    indices = set()
    
    def dfs(node):
        indices.add(node["cate_index"])
        for c in node["children"]:
            dfs(c)
    
    dfs(tree)
    return sorted(idx for idx in indices if idx != -1)


def hierarchical_based_metrics(pred_tree_nodes, true_tree_nodes):
    """
    Calculate Hierarchical metrics
    """

    least_common_ancestor = 0
    jaccard = 0
    hierarchical_precision = 0
    hierarchical_recall = 0

    intersection = pred_tree_nodes.intersection(true_tree_nodes)
    union = pred_tree_nodes.union(true_tree_nodes)
    
    jaccard += len(intersection) / len(union)
    hierarchical_precision += len(intersection) / len(pred_tree_nodes)
    hierarchical_recall += len(intersection) / len(true_tree_nodes)
        
    return jaccard, hierarchical_precision, hierarchical_recall

def dict_to_nx_tree(tree_dict):

    G = nx.DiGraph()  

    def add_nodes_edges(node, parent=None):
        node_id = node['cate_index']
        G.add_node(node_id, level=node['level'])
        
        if parent is not None:
            G.add_edge(parent, node_id)
        
        for child in node['children']:
            add_nodes_edges(child, parent=node_id)
    

    add_nodes_edges(tree_dict)
    return G

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
        tag_feats_path = 'hitag/data/frozen_tag_embedding/tag_des_embedding.pth'
        label_embed = nn.Parameter(torch.load(tag_feats_path,map_location='cpu').float())
        
        if args.model_type == "hitag":
            model = hitag(
                backbone=args.backbone,
                checkpoint=args.checkpoint,
                input_size=args.input_size,
                taglist=taglist,
                tag_des = tag_des,
                open_set=args.open_set,
                class_idxs=class_idxs,
                label_embed=label_embed
            )

        logits = torch.empty(len(imglist), len(taglist))
        pos = 0
        for imgs in tqdm(loader, desc="inference"):
            if args.model_type == "hitag":
                out = forward_hitag(model, imgs)

            bs = imgs.shape[0]
            logits[pos:pos+bs, :] = out.cpu()
            pos += bs

        # save logits, making threshold-tuning super fast
        torch.save(logits, logit_file)

    # filter with thresholds
    pred_tags = []
    for scores in logits.tolist():
        pred_tags.append([
            taglist[i] for i, s in enumerate(scores) if s >= thresholds[i]
        ])
    
    hire_gt_tree_path = 'hitag/data/hire_tree_file/object_structure.json'
    with open(hire_gt_tree_path, 'r') as f:
        hire_gt_tree = json.load(f)
    
    obj_tag2index_path = 'hitag/data/hire_tree_file/tag2idx_dict.json'
    with open(obj_tag2index_path, 'r') as f:
        obj_tag2index = json.load(f)

    total_tree_edit_distance = 0
    total_jaccard = 0
    total_hier_precision = 0
    total_hier_recall = 0
    total_val_size = len(pred_tags)
    
    for pred_tag, true_tree in zip(pred_tags, annot_file):
        pred_paths = build_category_paths(pred_tag, hire_gt_tree)
        
        pred_merged_tree = merge_paths_to_tree(pred_paths, obj_tag2index) 
        
        pred_all_indices = collect_all_indices_from_tree(pred_merged_tree)   
        
        true_all_indices = collect_all_indices_from_tree(true_tree)
        
        pred_tree_nx = dict_to_nx_tree(pred_merged_tree)
        true_tree_nx = dict_to_nx_tree(true_tree)
        
        tree_edit_dis = tree_edit_distance(pred_tree_nx, true_tree_nx)

        jaccard, hierarchical_precision, hierarchical_recall = hierarchical_based_metrics(
                    pred_all_indices, 
                    true_all_indices, 
                    )
        

        total_tree_edit_distance += tree_edit_dis
        total_jaccard += jaccard
        total_hier_precision += hierarchical_precision
        total_hier_recall += hierarchical_recall

    avg_tree_edit_distance = total_tree_edit_distance / total_val_size
    avg_jaccard = total_jaccard / total_val_size
    avg_hier_precision = total_hier_precision / total_val_size
    avg_hier_recall = total_hier_recall / total_val_size
    
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(
                f"Hierarchical evaluation: {args.dataset}, {len(pred_tags)} images, "
                f"{len(class_idxs)} classes \n[avg tree edit distance: {avg_tree_edit_distance:.3f}, "
                f"jaccard similarity: {avg_jaccard:.3f}, hierarchical precision: {avg_hier_precision:.3f}, "
                f"hierarchical recall: {avg_hier_recall:.3f}] "
            )
