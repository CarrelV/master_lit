import sys, os,copy
import collections
import torch
import torch_pruning as tp
import transformers
import config as CFG
from models_CLIP import CLIPMoco,get_classic_model

from tokenizer import get_tokenizer,get_feature_extractor
from pruning import pruning_BERT_without_residual,pruning_ViT_without_residual

from model_BERTLsT import *
from model_ViTLsT import *
from transformers import AutoTokenizer, BartModel,BartConfig
from fisher import compute_fisher
from dataloader import get_local_dataloader
from utils_models import modify_model_after_init,resume_model

if __name__ == "__main__":
    
    #print("BART")
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    #model = BartModel.from_pretrained("facebook/bart-base")
    #print(model.state_dict().keys())

    #bart_config = BartConfig.from_pretrained("facebook/bart-base")
    #print(bart_config)
    ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()


    reduction_factors = [8]
    #Temporary classic model for computing fisher:
    model = get_classic_model()
    #model = CLIPMoco()
    #print(text_config)
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)

    #print("Original model")
    #for n, p in model.named_parameters():

        
    #    print(n)


    importance_measure = compute_fisher(model, get_local_dataloader(dataset="flickr30k",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=8,shuffle=CFG.shuffle_train,split="train"), num_samples=CFG.samples_for_fisher)

    print("\n importance mesure \n \n")
    
    for name,tensor in importance_measure.items():
        break
        if torch.max(tensor) > 0:
            print(f"\n Name: {name}")
            print(f"tensor shape: {tensor.shape}")

    #print(model)
    #print("state dict")
    #print(model.state_dict().keys())
    print("\n Reduced model")
    for reduction_factor in reduction_factors:
        
        # pruned_state_dict = pruning_v1(model, reduction_factor)
        pruned_state_dict_text,pruned_idx_text = pruning_BERT_without_residual(model.text_encoder,tokenizer,reduction_factor,importance_measure=importance_measure)
        pruned_state_dict_img,pruned_idx_img = pruning_ViT_without_residual(model.image_encoder, feature_extractor, reduction_factor, importance_measure=importance_measure)
        

        '''print("\n pruned idx for text\n \n")
        count = 0
        for n,id in pruned_idx_text.items():
            
            count += 1
            print(f"Name: {n}")
            print(f"content: {id}")
            print(f"shape: {len(id)}")


            #if count > 10:
            #    break
            
        print("\n pruned idx for img\n \n")
        count = 0
        for n,id in pruned_idx_img.items():
            
            count += 1
            print(f"Name: {n}")
            print(f"content: {id}")
            print(f"shape: {len(id)}")'''


    model = CLIPMoco()

    check = False
    for n,p in model.text_encoder.named_parameters():

        if ("downsampler" in n) and ("weight" in n):
            

            print(f"Name: {n}")

            print(f"original weight shape: {p.shape}")
            infer_n = n.split(".")
            number = infer_n[1]
            list_of_index = pruned_idx_text[f"model.encoder.layer.{number}.output.LayerNorm.weight"]
            print(f"list of index: {list_of_index}")

            new_weights = torch.zeros(p.shape)

            print(f"new weights size: {new_weights.shape}")
            for i,index in enumerate(list_of_index):
                new_weights[i,index] = 1

            print("new weights")
            print(new_weights)

            break


    '''print("Not pruned")
    for n, p in model.image_encoder.named_parameters():

        if n not in pruned_state_dict_img:
            print(n)

    print("\n pruned state for text\n \n")

    print(pruned_state_dict_text.keys())

    print("Not pruned")
    for n, p in model.text_encoder.named_parameters():

        if n not in pruned_state_dict_text:
            print(n)'''

    '''
    side_config = copy.deepcopy(text_config)

    side_config.intermediate_size = side_config.intermediate_size // CFG.reduction_factor
    side_config.hidden_size = side_config.hidden_size // CFG.reduction_factor

    new_model = BertLSTModel(config=side_config)
    print(new_model)'''