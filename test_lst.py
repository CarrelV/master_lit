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

from model_BERTLsT import BertLSTModel

from transformers import AutoTokenizer, BertModel,BertConfig

from utils_models import modify_text_model_after_init

if __name__ == "__main__":
    
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_config = BertConfig.from_pretrained(CFG.text_model_name)
    model = CLIPMoco().to(device)
    

    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    text = tokenizer("Hello, my dog is cute", return_tensors="pt")

    dummy_image = np.zeros((256,256,3), np.uint8)
    image = feature_extractor(dummy_image,return_tensors="pt")

    
    model = modify_text_model_after_init(model,tokenizer)
#print(model)
    
    #model.copy_pretrain_weight(tokenizer,8)

    #outputs = model(image["pixel_values"],text)


    #print(outputs)
#   