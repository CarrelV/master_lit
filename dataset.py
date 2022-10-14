import re
import json
import os
from PIL import Image
import random 

import torch
from torch.utils.data import  Dataset
from torchvision.datasets.utils import download_url

import config as CFG

def pre_caption(caption,max_words=CFG.max_length):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class flickr30k(Dataset):
    def __init__(self, tokenizer,transform,image_root, ann_root, split, max_words=CFG.max_length, prompt=CFG.prompt):        
        '''
        image_root (string): Root directory of images (e.g. data/)
        ann_root (string): directory to store the annotation file
        split (string): one of "train" or "test"
        '''        
      

        self.tokenizer = tokenizer

        self.split = split
        assert self.split in ("train","val","test")

        if self.split == "train":
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json"
            filename = "flickr30k_train.json"
        elif self.split == "val":
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json"
            filename = "flickr30k_val.json"
        else:
            url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json"
            filename = "flickr30k_test.json"

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        
        self.img_ids = {} 
        
        ## Totally bad way to merge the caption, cba to redo

        if self.split == "train":
            n = 0
            last_img_id = None
            current_img_id = None
            for ann in self.annotation:
                current_img_id = ann["image_id"]

                if current_img_id != last_img_id:
                    self.img_ids[n] = ann
                    self.img_ids[n]["caption"] = [self.img_ids[n]["caption"]]
                    last_img_id = current_img_id
                    n += 1
                else: 
                    ls = self.img_ids[n-1]["caption"]
                    ls.append(ann["caption"])               
                    self.img_ids[n-1]["caption"] = ls
                
                
        else:
            n = 0
            for ann in self.annotation:
                self.img_ids[n] = ann
                n += 1

        self.annotation = None
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):    
        
        item = self.img_ids[index]
        
        image_path = os.path.join(self.image_root,item['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(random.choice(item['caption']), self.max_words)
        
        caption_encoded = self.tokenizer(caption,padding="max_length",max_length=self.max_words)

        return {"image" :image, "input_ids": torch.as_tensor(caption_encoded["input_ids"]), "attention_mask": torch.as_tensor(caption_encoded["attention_mask"])}

def get_dataset(tokenizer,transform,split):
    
    return flickr30k(tokenizer=tokenizer,transform=transform,image_root=CFG.image_root,ann_root=CFG.ann_root,split=split)

    
