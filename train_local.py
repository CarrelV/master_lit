import config as CFG
from dataloader import get_dataloader,get_local_dataloader
from tokenizer import get_tokenizer,get_feature_extractor
from models_CLIP import CLIPMoco
from losses import CLIPMoCOLoss
from training_local import train_one_epoch, valid_one_epoch,get_lr
from utils import setup,cleanup
from utils_models import modify_text_model_after_init,resume_model

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging

from fisher import compute_fisher

## Seems to be working when testing locally with GPU = 1. Maybe hacky/wrong fix? Will test tomorrow on the cluster
def main():
    logging.set_verbosity_error()
    wandb.init(project="Master Thesis Project",
           config={
               "batch_size": CFG.batch_size,
               "dataset": "flickr30k",
           },
           group="Baselines")

    # setup the process groups
    #setup(rank, world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    dataloader_train = get_local_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="train")
    dataloader_valid = get_local_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="val")

    number_of_step_per_epoch = len(dataloader_train)
    
    loss_fn = CLIPMoCOLoss().to(device)
    model = CLIPMoco()

    # copy the pruned weights of the main text to the side LST text network
    if CFG.side_text_weights_copy:
       
        importance_measure = compute_fisher(model, get_local_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=1,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)
        
        model = modify_text_model_after_init(model,tokenizer,importance_measure)



    model.to(device)

    
    resume_model(model)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    
    
    #Parameter
    params = []
    if CFG.text_backbone_finetune:
        params.append({"params" : model.text_encoder.parameters(), "lr" : CFG.text_encoder_lr,"weight_decay": 0.})
    if CFG.image_backbone_finetune:
        params.append({"params" : model.image_encoder.parameters(), "lr" : CFG.image_encoder_lr,"weight_decay": 0.})
    params.append({"params" : model.text_projection.parameters(), "lr" : CFG.text_head_lr,"weight_decay": 0.})
    params.append({"params" : model.image_projection.parameters(), "lr" : CFG.image_head_lr,"weight_decay": 0.})


    #Optimizer
    optimizer = torch.optim.AdamW(params)
    #Learning rate
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=CFG.warming_epochs,num_training_steps=CFG.epochs)


    
    best_loss = float('inf')

    for epoch in range(CFG.epochs):
       
        print(f"Epoch: {epoch + 1}")

        ## TRAINING
        model.train()
        
        train_loss = train_one_epoch(model, loss_fn, dataloader_train, optimizer,device)
            
        ## VALIDATION
        model.eval()

        with torch.no_grad():
            valid_loss = valid_one_epoch(model,loss_fn,dataloader_valid,device)
        
        
        if valid_loss.avg_loss < best_loss:
            best_loss = valid_loss.avg_loss

            # Save the backbone
            if CFG.text_backbone_finetune:
                torch.save(model.text_encoder.state_dict(), f"weights/{CFG.configuration}_text_enc_best_{CFG.training_run_number}.pt")
            if CFG.image_backbone_finetune:
                torch.save(model.image_encoder.state_dict(), f"weights/{CFG.configuration}_img_enc_best_{CFG.training_run_number}.pt")
            # Save the two heads
            torch.save(model.text_projection.state_dict(), f"weights/{CFG.configuration}_text_proj_best_{CFG.training_run_number}.pt")
            torch.save(model.image_projection.state_dict(), f"weights/{CFG.configuration}_img_proj_best_{CFG.training_run_number}.pt")
        
        
        wandb.log({"Training loss": train_loss.avg_loss, "Validation loss" : valid_loss.avg_loss}, commit = False )

        if CFG.text_backbone_finetune:
            wandb.log({"Text Encoder lr" : lr_scheduler.get_last_lr()[0]},commit = False)

        wandb.log({"Text Projection lr" : lr_scheduler.get_last_lr()[-2], "Image Projection lr": lr_scheduler.get_last_lr()[-1]})

        lr_scheduler.step()

    
    cleanup()




if __name__ == "__main__":

    main()
