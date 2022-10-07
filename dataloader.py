import re
import json
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url



def pre_caption(caption,max_words=128):
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
    def __init__(self, transform, image_root, ann_root, split, max_words=128, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. data/)
        ann_root (string): directory to store the annotation file
        split (string): one of "train" or "test"
        '''        
        train = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json'
        test = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'
        filename = 'flickr30k_train.json'

        self.split = split
        assert self.split in ("train","test")

        if self.split == "train":
            url = train
        else:
            url = test

        download_url(url,ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words)
        
        #return image, caption, self.img_ids[ann['image_id']] 
        return image, caption

def collate_custom(batch):
 
    imgs = torch.stack([item[0] for item in batch])
    caps = [item[1] for item in batch]
 
    return imgs, caps