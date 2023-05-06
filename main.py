import config as CFG
from dataloader import get_dataloader
from tokenizer import get_tokenizer,get_feature_extractor
from models_CLIP import CLIPMoco, get_classic_model
from losses import CLIPMoCOLoss
from training import train_one_epoch, valid_one_epoch,get_lr
from utils import setup,cleanup
from utils_models import modify_model_after_init,resume_model

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging

from fisher import compute_fisher



## Seems to be working when testing locally with GPU = 1. Maybe hacky/wrong fix? Will test tomorrow on the cluster
def main(rank,world_size):
    logging.set_verbosity_error()
    wandb.init(project="Master Thesis Project",
           config={
               "batch_size": CFG.batch_size,
               "dataset": CFG.dataset,
           },
           group="Baselines")

    # setup the process groups

    setup(rank, world_size)
    
    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    dataloader_train = get_dataloader(dataset=CFG.dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,rank=rank,world_size=world_size,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="train")
    dataloader_valid = get_dataloader(dataset=CFG.dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,rank=rank,world_size=world_size,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="val")

    number_of_step_per_epoch = len(dataloader_train)
    
    model = CLIPMoco()
    
    loss_fn = CLIPMoCOLoss().to(rank)
    # copy the pruned weights of the main text to the side LST text network
    
    
    
    if CFG.side_text_weights_copy or CFG.side_image_weights_copy:
        print("Starting copying the weights to the pruned side network")
        # Create a simple model without the ladder to computer the fisher, then write the new over
        importance_measure = compute_fisher(model, get_dataloader(dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,rank=rank,world_size=world_size,batch_size=1,shuffle=CFG.shuffle_train,num_workers=CFG.num_workers,split="train"), num_samples=CFG.samples_for_fisher)
        print("fisher importance measure computed")

        # Copy the pruned weights
        model = modify_model_after_init(model,tokenizer,feature_extractor,importance_measure)
        print("Finish copying the weights to the pruned side network")
        importance_measure = None

    
        
    model.to(rank)
    model = DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=CFG.find_unused_param)

    resume_model(model)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    
    
    
    #Parameter
    params = []
    if CFG.text_backbone_finetune:
        params.append({"params" : [p for n,p in model.module.text_encoder.named_parameters() if (("side_encoder" in n) or ("downsampler" in n)) and (p.requires_grad)], "lr" : CFG.text_encoder_lr})
        params.append({"params" : [p for n,p in model.module.text_encoder.named_parameters() if ("side_encoder" not in n) and ("downsampler" not in n) and (p.requires_grad)], "lr" : CFG.text_head_lr})
        #params.append({"params" : model.module.text_encoder.parameters(), "lr" : CFG.text_encoder_lr})
    if CFG.image_backbone_finetune:
        params.append({"params" : [p for n,p in model.module.image_encoder.named_parameters() if (("side_encoder" in n) or ("downsampler" in n)) and (p.requires_grad)], "lr" : CFG.image_encoder_lr})
        params.append({"params" : [p for n,p in model.module.image_encoder.named_parameters() if ("side_encoder" not in n) and ("downsampler" not in n) and (p.requires_grad)], "lr" : CFG.image_head_lr})
        #params.append({"params" : model.module.image_encoder.parameters(), "lr" : CFG.image_encoder_lr})
    params.append({"params" : model.module.text_projection.parameters(), "lr" : CFG.text_head_lr})
    params.append({"params" : model.module.image_projection.parameters(), "lr" : CFG.image_head_lr})


    #Optimizer
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    #Learning rate
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=CFG.warming_epochs,num_training_steps=CFG.epochs)

    
    best_loss = float('inf')

    for epoch in range(CFG.epochs):
       
        print(f"Epoch: {epoch + 1}")

        ## TRAINING
        model.train()
        
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader_train.sampler.set_epoch(epoch)
        dataloader_valid.sampler.set_epoch(epoch)


        train_loss = train_one_epoch(model, loss_fn, dataloader_train, optimizer,rank)
            
        ## VALIDATION
        model.eval()

        with torch.no_grad():
            valid_loss = valid_one_epoch(model,loss_fn,dataloader_valid,rank)
        
        

        if valid_loss.avg_loss < best_loss:
            best_loss = valid_loss.avg_loss

            # Save the backbone
            if CFG.text_backbone_finetune:
                torch.save(model.module.text_encoder.state_dict(), f"weights/{CFG.configuration}_text_enc_best_{CFG.training_run_number}.pt")
            if CFG.image_backbone_finetune:
                torch.save(model.module.image_encoder.state_dict(), f"weights/{CFG.configuration}_img_enc_best_{CFG.training_run_number}.pt")
            # Save the two heads
            torch.save(model.module.text_projection.state_dict(), f"weights/{CFG.configuration}_text_proj_best_{CFG.training_run_number}.pt")
            torch.save(model.module.image_projection.state_dict(), f"weights/{CFG.configuration}_img_proj_best_{CFG.training_run_number}.pt")
        
        wandb.log({"Training loss": train_loss.avg_loss, "Validation loss" : valid_loss.avg_loss}, commit = False )

        if CFG.text_backbone_finetune:
            wandb.log({"Text Encoder lr" : lr_scheduler.get_last_lr()[0]},commit = False)
        if CFG.image_backbone_finetune:
            wandb.log({"Image Encoder lr" : lr_scheduler.get_last_lr()[1]},commit = False)

        wandb.log({"Text Projection lr" : lr_scheduler.get_last_lr()[-2], "Image Projection lr": lr_scheduler.get_last_lr()[-1]})

        lr_scheduler.step()

    print("Finish training")
    cleanup()




if __name__ == "__main__":

    
    # world_size is the number of GPU available
    world_size = CFG.gpu_number   
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
