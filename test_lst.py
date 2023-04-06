import sys, os,copy
import collections
import torch
import torch_pruning as tp
import transformers
import config as CFG
from models_CLIP import CLIPMoco
import numpy as np
from transformers import logging

from tokenizer import get_tokenizer,get_feature_extractor
from pruning import pruning_BERT_without_residual

from fisher import compute_fisher
from dataloader import get_local_dataloader

from utils_models import modify_text_model_after_init,resume_model

if __name__ == "__main__":
    
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CLIPMoco().to(device)
    

    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    print(feature_extractor)
    text = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)

    dummy_image = np.zeros((256,256,3), np.uint8)
    image = feature_extractor(dummy_image,return_tensors="pt").to(device)
    print("Hallo")
    if CFG.side_text_weights_copy:
       
        importance_measure = compute_fisher(model, get_local_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=1,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)
        
        model = modify_text_model_after_init(model,tokenizer,importance_measure,device)


    resume_model(model)


    print("Finish")
    #model.copy_pretrain_weight(tokenizer,8)

    outputs = model(image["pixel_values"],text)

    print(outputs)
    #print(outputs)
#   