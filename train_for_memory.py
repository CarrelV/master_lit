import config as CFG
from dataloader import get_local_dataloader
from tokenizer import get_tokenizer,get_feature_extractor
from models_CLIP import CLIPMoco, get_classic_model
from losses import CLIPMoCOLoss
from training_for_memory import train_one_epoch, valid_one_epoch,get_lr
from utils import setup,cleanup
from utils_models import modify_model_after_init,resume_model

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import logging

from fisher import compute_fisher

from metric import imagenet_0shot,i2t_t2i_retrieval
import csv
data_to_save = []


def main():
    logging.set_verbosity_error()
    

    # setup the process groups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_to_save.append(CFG.run_info)
    data_to_save.append(CFG.text_tower_name)
    data_to_save.append(CFG.text_head_name)
    data_to_save.append(CFG.image_tower_name)
    data_to_save.append(CFG.image_head_name)
    # prepare the dataloader
    
    print("prepare the dataloader")

    #print(f"Beginning memory: {torch.cuda.memory_allocated(rank)}")

    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    dataloader_train = get_local_dataloader(dataset=CFG.dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="train")
    dataloader_valid = get_local_dataloader(dataset=CFG.dataset,tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="val")

 
    number_of_step_per_epoch = len(dataloader_train)
    
    print("prepare the model")
    mem_before_model = torch.cuda.memory_allocated()
    loss_fn = CLIPMoCOLoss().to(device)
    model = CLIPMoco()
    # copy the pruned weights of the main text to the side LST text network
    
    
    if CFG.side_text_weights_copy or CFG.side_image_weights_copy:
        print("Starting copying the weights to the pruned side network")
        # Create a simple model without the ladder to computer the fisher, then write the new over
        importance_measure = compute_fisher(model, get_local_dataloader(dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=1,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)
        print("fisher importance measure computed")

        # Copy the pruned weights
        model = modify_model_after_init(model,tokenizer,feature_extractor,importance_measure)
        print("Finish copying the weights to the pruned side network")
        importance_measure = None

    
        
    model.to(device)

    resume_model(model)

    text_tower_trainable_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    text_tower_total_params = sum(p.numel() for p in model.text_encoder.parameters())
    text_head_trainable_params = sum(p.numel() for p in model.text_projection.parameters() if p.requires_grad)
    text_head_total_params = sum(p.numel() for p in model.text_projection.parameters())
    image_tower_trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    image_tower_total_params = sum(p.numel() for p in model.image_encoder.parameters())
    image_head_trainable_params = sum(p.numel() for p in model.image_projection.parameters() if p.requires_grad)
    image_head_total_params = sum(p.numel() for p in model.image_projection.parameters())

    data_to_save.append(text_tower_total_params)
    data_to_save.append(text_head_total_params)
    data_to_save.append(image_tower_total_params)
    data_to_save.append(image_head_total_params)
    data_to_save.append(text_tower_trainable_params)
    data_to_save.append(text_head_trainable_params)
    data_to_save.append(image_tower_trainable_params)
    data_to_save.append(image_head_trainable_params)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    data_to_save.append(mem_before_model)

    memory_after_model = torch.cuda.memory_allocated() 
    data_to_save.append(memory_after_model)
    
    #Parameter
    params = []
    if CFG.text_backbone_finetune:
        #params.append({"params" : [p for n,p in model.module.text_encoder.named_parameters() if (("side_encoder" in n) or ("downsampler" in n)) and (p.requires_grad)], "lr" : CFG.text_encoder_lr})
        #params.append({"params" : [p for n,p in model.module.text_encoder.named_parameters() if ("side_encoder" not in n) and ("downsampler" not in n) and (p.requires_grad)], "lr" : CFG.text_head_lr})
        params.append({"params" : model.text_encoder.parameters(), "lr" : CFG.text_encoder_lr})
    if CFG.image_backbone_finetune:
        #params.append({"params" : [p for n,p in model.module.image_encoder.named_parameters() if (("side_encoder" in n) or ("downsampler" in n)) and (p.requires_grad)], "lr" : CFG.image_encoder_lr})
        #params.append({"params" : [p for n,p in model.module.image_encoder.named_parameters() if ("side_encoder" not in n) and ("downsampler" not in n) and (p.requires_grad)], "lr" : CFG.image_head_lr})
        params.append({"params" : model.image_encoder.parameters(), "lr" : CFG.image_encoder_lr})
    params.append({"params" : model.text_projection.parameters(), "lr" : CFG.text_head_lr})
    params.append({"params" : model.image_projection.parameters(), "lr" : CFG.image_head_lr})


    #Optimizer
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    #Learning rate
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=CFG.warming_epochs*number_of_step_per_epoch,num_training_steps=CFG.epochs*number_of_step_per_epoch)



    best_loss = float("inf")

    
    for epoch in range(CFG.epochs):
       
        print(f"Epoch: {epoch + 1}")
        ## TRAINING
        model.train()
    
        


        train_loss,data_returned = train_one_epoch(model, loss_fn, dataloader_train, optimizer,lr_scheduler,device)
        ## VALIDATION
        model.eval()


        for item in data_returned:
            data_to_save.append(item)
        #with torch.no_grad():
        #    valid_loss = valid_one_epoch(model,loss_fn,dataloader_valid,device)



        break


        
            
        
       
        #lr_scheduler.step()

    with open("memory.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_to_save)
    print("Finish training")
    




if __name__ == "__main__":

    
    main()
