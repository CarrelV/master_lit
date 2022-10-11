import config as CFG
from dataloader import get_dataloader, get_DDP_dataloader
from tokenizer import get_tokenizer
from models import CLIPModel
from losses import CLIPLoss
from training import train_one_epoch
from utils import setup,cleanup

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import torch
import itertools

def main():

    
    
    tokenizer = get_tokenizer(CFG.text_model_name)
    dataloader_train = get_dataloader(tokenizer=tokenizer,batch_size=CFG.batch_size,shuffle=CFG.shuffle,num_workers=CFG.num_workers,split=CFG.split)


    model = CLIPModel().to(CFG.device)
    loss_fn = CLIPLoss()
    
    if CFG.trainable == False:
        params = [
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    else: 
        params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    

    best_loss = float('inf')

    for epoch in range(CFG.epochs):
        
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_one_epoch(model, loss_fn, dataloader_train, optimizer)
        
        current_loss = train_loss.avg_loss.item()

        if current_loss < best_loss:
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(train_loss.avg_loss)



def main_DDP(rank,world_size):

    # setup the process groups
    setup(rank, world_size)
    
    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    dataloader_train = get_DDP_dataloader(tokenizer=tokenizer,rank=rank,world_size=world_size,batch_size=CFG.batch_size,shuffle=CFG.shuffle,num_workers=CFG.num_workers,split=CFG.split)


    model = CLIPModel().to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    
    #Loss fct
    loss_fn = CLIPLoss()
    
    #Parameter
    if CFG.trainable == False:
        params = [
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    else: 
        params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    #Optimizer
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    #Learning rate
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    
    best_loss = float('inf')

    for epoch in range(CFG.epochs):
       
        print(f"Epoch: {epoch + 1}")
        model.train()
        
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader_train.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, loss_fn, dataloader_train, optimizer)    
        
        current_loss = train_loss.avg_loss.item()

        if current_loss < best_loss:
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(train_loss.avg_loss)

    cleanup()












if __name__ == "__main__":

    wandb.init(project="master_test_1",
           config={
               "batch_size": CFG.batch_size,
               "learning_rate": CFG.head_lr,
               "dataset": "flickr30k",
           })
    
    
    # suppose we have 2 gpus
    world_size = 2    
    mp.spawn(
        main_DDP,
        args=(world_size),
        nprocs=world_size
    )
    #main()