## Okay working

from dataset import get_dataset
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


transform_train = transforms.Compose([
            transforms.RandomCrop(224,pad_if_needed=True),
            transforms.RandomHorizontalFlip(), 
            ])

transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            ])

transform_ImageNet = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385), 
                                 (0.285, 0.277, 0.286))
            ])

            
# Not used since I switched to a dict for the DataSet 
def collate_custom(batch):
 
    imgs = torch.stack([item["image"] for item in batch])
    
    inputs_id = torch.stack([item["input_ids"] for item in batch])
    mask = torch.stack([item["attention_mask"] for item in batch])
 
    return imgs, inputs_id,mask

def get_dataloader(tokenizer,feature_extractor,batch_size,shuffle,num_workers,split):
    
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if split == "train":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    elif split == "val":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    elif split == "test":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test")
        return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    else:
        print("Wrong split")

def get_DDP_dataloader(tokenizer,feature_extractor,rank,world_size,batch_size,shuffle,num_workers,split,pin_memory=False):
    
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if split == "train":
        
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=False,num_workers=num_workers,sampler=sampler)
    
    elif split == "val":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="val")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=False,num_workers=num_workers,sampler=sampler)
    
    elif split == "test":
        dataset = get_dataset(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_test,split="test")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        return DataLoader(dataset=dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=False,num_workers=num_workers,sampler=sampler)
    else:
        print("Wrong split")