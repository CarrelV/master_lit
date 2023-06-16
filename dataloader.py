## Okay working

from dataset import get_dataset
import os
from PIL import Image
import webdataset as wds


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


transform_train = transforms.Compose([
            transforms.Resize(256,interpolation=Image.BICUBIC),
            transforms.RandomCrop(224,pad_if_needed=True),
            transforms.RandomHorizontalFlip(), 
            ])

transform_test = transforms.Compose([
            transforms.Resize(256,interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            ])





def get_dataloader(dataset,tokenizer,feature_extractor,rank,world_size,batch_size,shuffle,num_workers,split,pin_memory=False):
    
    if split == "train":
        
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train",rank=rank)
        
    elif split == "val":
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val",rank=rank)
    
    elif split == "test":
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test",rank=rank)
    else:
        print("Wrong split")



    if dataset == "cc3m":
        dataloader = wds.WebLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
        return DataLoader(dataloader,batch_size=None)

    else:   
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,num_workers=num_workers,sampler=sampler,drop_last=True)
    


def get_local_dataloader(dataset,tokenizer,feature_extractor,batch_size,shuffle,split):
    
    
    if split == "train":
        
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")
        
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)
    
    elif split == "val":
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)
    
    elif split == "test":
        dataset = get_dataset(dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)
    else:
        print("Wrong split")