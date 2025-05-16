import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption
import os,glob

import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

def extract_level_pairs(tree):

    pairs = []
    
    def traverse(node):
        for child in node.get("children", []):
            pairs.append((node["cate_index"], child["cate_index"]))
            traverse(child)
    
    traverse(tree)
    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(pairs, dtype=torch.int64)


def collate_fn(batch):

    padded_batch = pad_sequence(batch, batch_first=True, padding_value=-1)
    return padded_batch

class pretrain_obj_dataset(Dataset):
    def __init__(self, ann_file, transform, class_num = 3333, root = ''): 

        self.ann = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann += ann
            
        self.transform = transform
        self.class_num = class_num
        self.root = root

    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]   

        # else: 
        img_dir = '/path/to/cc3m/img_file/images'
        img_file = ann['image_path'].split('/')[-1]
        image_path_use = os.path.join(img_dir, img_file)
        
        image = Image.open(image_path_use).convert('RGB')   
        image = self.transform(image)
        
        tag_tree = torch.tensor(ann['union_label_tree'], dtype=torch.int64)


        if ann.get('union_label_tree_id') is not None:
            num = ann['union_label_tree_id'] 
            image_tag = np.zeros([self.class_num]) 
            image_tag[num] = 1 
            image_tag = torch.tensor(image_tag, dtype = torch.long)
        else:
            image_tag = None

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index],30)


        return image, caption, image_tag, tag_tree
    

class finetune_obj_dataset(Dataset):
    def __init__(self, ann_file, transform, transform_224, class_num = 4585, root = ''): 

        self.ann = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann += ann
            
        self.transform = transform
        self.transform_224 = transform_224
        self.class_num = class_num
        self.root = root

    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]   

        img_dir = '/path/to/finetune_dataset'
        img_file = ann['image_path']
        image_path_use = os.path.join(img_dir, img_file)
        
        image = Image.open(image_path_use).convert('RGB')   
        image = self.transform(image)

        image_224 = Image.open(image_path_use).convert('RGB')  
        image_224 = self.transform_224(image_224)
        
        tag_tree = torch.tensor(ann['union_label_tree'], dtype=torch.int64)

        # required for tag2text support
        if ann.get('union_label_tree_id') is not None:
            num = ann['union_label_tree_id'] 
            image_tag = np.zeros([self.class_num]) 
            image_tag[num] = 1 
            image_tag = torch.tensor(image_tag, dtype = torch.long)
        else:
            image_tag = None

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index],30)

        num = ann['parse_label_id'][caption_index]


        return image, image_224, caption, image_tag, tag_tree
    
