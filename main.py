import config as CFG
from dataloader import get_dataloader, get_DDP_dataloader
from tokenizer import get_tokenizer,get_feature_extractor
from models import CLIPProjMoco, CLIPProjection
from losses import CLIPLoss, CLIPMoCOLoss
from training import train_one_epoch, valid_one_epoch,train_one_MOCO_epoch
from utils import setup,cleanup

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import torch
import itertools
from transformers import logging


def main():

    logging.set_verbosity_error()
    wandb.init(project="master_test_1",
           config={
               "batch_size": CFG.batch_size,
               "learning_rate": CFG.head_lr,
               "dataset": "flickr30k",
           },
           group="MOCO")
    
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    dataloader_train = get_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="train")
    dataloader_valid = get_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="val")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CFG.model_used == "classic":
        model = CLIPProjection().to(device)
        loss_train = CLIPLoss().to(device)

    
    elif CFG.model_used == "moco":
        model = CLIPProjMoco().to(device)
        loss_train = CLIPMoCOLoss().to(device)
    
    elif CFG.model_used == "ape":
        print("Model not implemented yet")
        return 
    
    
    else:
        print("Model not implemented yet")
        return

    loss_validation = CLIPLoss().to(device)
    
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max)
    
    

    best_loss = float('inf')

    for epoch in range(CFG.epochs):
        
        print(f"Epoch: {epoch + 1}")
        
        ## TRAINING
        model.train()

        if CFG.model_used == "classic":
            train_loss = train_one_epoch(model, loss_train, dataloader_train, optimizer,device)

        elif CFG.model_used == "moco":
            train_loss = train_one_MOCO_epoch(model, loss_train, dataloader_train, optimizer,device)
        
        elif CFG.model_used == "ape":
            print("Model not implemented yet")
            return 
        
        else:
            print("Model not implemented yet")
            return

        
        ## VALIDATION  
        model.eval()

        with torch.no_grad():
            valid_loss = valid_one_epoch(model,valid_loss,dataloader_valid,device)

        if valid_loss.avg_loss < best_loss:
            best_loss = valid_loss.avg_loss

            if CFG.trainable:
                # Save the two towers
                torch.save(model.image_encoder.state_dict(), f"weights/{CFG.model_used}_img_enc_best_{CFG.training_run_number}.pt")
                torch.save(model.text_encoder.state_dict(), f"weights/{CFG.model_used}_text_enc_best_{CFG.training_run_number}.pt")
                # And the two projection heads
                torch.save(model.image_projection.state_dict(), f"weights/{CFG.model_used}_img_proj_best_{CFG.training_run_number}.pt")
                torch.save(model.text_projection.state_dict(), f"weights/{CFG.model_used}_text_proj_best_{CFG.training_run_number}.pt")

            else:
                #Only save projection heads
                torch.save(model.image_projection.state_dict(), f"weights/{CFG.model_used}_img_proj_best_{CFG.training_run_number}.pt")
                torch.save(model.text_projection.state_dict(), f"weights/{CFG.model_used}_text_proj_best_{CFG.training_run_number}.pt")

                #print("Saved Best Model!")
        

        
        lr_scheduler.step()




## Seems to be working when testing locally with GPU = 1. Maybe hacky/wrong fix? Will test tomorrow on the cluster
def main_DDP(rank,world_size):

    logging.set_verbosity_error()
    wandb.init(project="master_test_1",
           config={
               "batch_size": CFG.batch_size,
               "learning_rate": CFG.head_lr,
               "dataset": "flickr30k",
           },
           group="MOCO")

    # setup the process groups
    setup(rank, world_size)
    
    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    dataloader_train = get_DDP_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,rank=rank,world_size=world_size,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="train")
    dataloader_valid = get_DDP_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,rank=rank,world_size=world_size,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="val")

    if CFG.model_used == "classic":
        model = CLIPProjection().to(rank)
        loss_train = CLIPLoss().to(rank)

    
    elif CFG.model_used == "moco":
        model = CLIPProjMoco().to(rank)
        loss_train = CLIPMoCOLoss().to(rank)
    
    elif CFG.model_used == "ape":
        print("Model not implemented yet")
        return 
    
    
    else:
        print("Model not implemented yet")
        return

    loss_validation = CLIPLoss().to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=False)
    
    
    #Parameter
    if CFG.trainable == False:
        params = [
            {"params": itertools.chain(
                model.module.image_projection.parameters(), model.module.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    else: 
        params = [
            {"params": model.module.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.module.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.module.image_projection.parameters(), model.module.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]

    #Optimizer
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    #Learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max)

    
    best_loss = float('inf')

    for epoch in range(CFG.epochs):
       
        print(f"Epoch: {epoch + 1}")

        ## TRAINING
        model.train()
        
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader_train.sampler.set_epoch(epoch)
        dataloader_valid.sampler.set_epoch(epoch)


        if CFG.model_used == "classic":
            train_loss = train_one_epoch(model, loss_train, dataloader_train, optimizer,rank)

        elif CFG.model_used == "moco":
            train_loss = train_one_MOCO_epoch(model, loss_train, dataloader_train, optimizer,rank)
        
        elif CFG.model_used == "ape":
            print("Model not implemented yet")
            return 
        
        else:
            print("Model not implemented yet")
            return

        
        ## VALIDATION
        model.eval()

        with torch.no_grad():
            valid_loss = valid_one_epoch(model,valid_loss,dataloader_valid,rank)

       

        if valid_loss.avg_loss < best_loss:
            best_loss = valid_loss.avg_loss
            if CFG.trainable:
                # Save the two towers
                torch.save(model.image_encoder.state_dict(), f"weights/{CFG.model_used}_img_enc_best_{CFG.training_run_number}.pt")
                torch.save(model.text_encoder.state_dict(), f"weights/{CFG.model_used}_text_enc_best_{CFG.training_run_number}.pt")
                # And the two projection heads
                torch.save(model.image_projection.state_dict(), f"weights/{CFG.model_used}_img_proj_best_{CFG.training_run_number}.pt")
                torch.save(model.text_projection.state_dict(), f"weights/{CFG.model_used}_text_proj_best_{CFG.training_run_number}.pt")

            else:
                #Only save projection heads
                torch.save(model.image_projection.state_dict(), f"weights/{CFG.model_used}_img_proj_best_{CFG.training_run_number}.pt")
                torch.save(model.text_projection.state_dict(), f"weights/{CFG.model_used}_text_proj_best_{CFG.training_run_number}.pt")

                #print("Saved Best Model!")
        
        lr_scheduler.step()


    cleanup()








if __name__ == "__main__":

    
    
    '''
    
    # world_size is the number of GPU available'''
    world_size = CFG.gpu_number   
    mp.spawn(
        main_DDP,
        args=(world_size,),
        nprocs=world_size
    )
    
    # Use for single GPU training

    #main()