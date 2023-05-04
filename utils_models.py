
from pruning import pruning_BERT_without_residual,pruning_ViT_without_residual
import config as CFG

import torch








# 
def modify_model_after_init(model,tokenizer,feature_extractor,importance_measure):

    step = CFG.ladder_reduction_factor
    
    initial_gap = CFG.ladder_initial_gap
    


    if CFG.side_text_weights_copy:
        print("starting with text encoder")
        side_state_dict_text,pruned_idx_text = pruning_BERT_without_residual(model.text_encoder,tokenizer,CFG.reduction_factor,importance_measure)

        print("pruning of text finished, copying ...")
        for n,p in model.text_encoder.named_parameters():
         
            #Copy the side encoder weights
            if "side_encoder" in n:
                
                infer_n = n.split(".")
                infer_n[0] = "encoder.layer"

                infer_n[1] = str(initial_gap + int(infer_n[1]) * step)
                infer_n = ".".join(infer_n)            

                p.data.copy_(side_state_dict_text[infer_n])

            #Init the downsampler as identity of pruned ladder
            if ("downsampler" in n) and ("weight" in n):

                infer_n = n.split(".")
                number = initial_gap + int(infer_n[1]) * step
                
                list_of_index = pruned_idx_text[f"encoder.layer.{number}.output.LayerNorm.weight"]

                new_weights = torch.zeros(p.shape)

                for i,index in enumerate(list_of_index):
                    new_weights[i,index] = 1

                p.data.copy_(new_weights)

            

    if CFG.side_image_weights_copy:

        print("starting with image encoder")

        side_state_dict_image,pruned_idx_img = pruning_ViT_without_residual(model.image_encoder,feature_extractor,CFG.reduction_factor,importance_measure)
        print("pruning of image finished, copying ...")

        for n,p in model.image_encoder.named_parameters():
            
            #Copy the side encoder weights
            if "side_encoder" in n:
                
                infer_n = n.split(".")
                infer_n[0] = "encoder.layer"

                infer_n[1] = str(initial_gap + int(infer_n[1]) * step)
                infer_n = ".".join(infer_n)            

                p.data.copy_(side_state_dict_image[infer_n])

            #Init the downsampler as identity of pruned ladder
            if ("downsampler" in n) and ("weight" in n):

                infer_n = n.split(".")
                number = initial_gap + int(infer_n[1]) * step
                list_of_index = pruned_idx_img[f"encoder.layer.{number}.layernorm_after.weight"]

                new_weights = torch.zeros(p.shape)

                for i,index in enumerate(list_of_index):
                    new_weights[i,index] = 1

                p.data.copy_(new_weights)

    print("Finish post init of model")
    
    return model

def resume_model(model):
    
    print(f"Config: {CFG.configuration}, text model size: {CFG.text_model_size}, image model size: {CFG.image_model_size}")

    trainable_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.text_encoder.parameters())

    print(f"For the Text encoder:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.text_projection.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.text_projection.parameters())

    print(f"For the Text head:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_encoder.parameters())

    print(f"For the Image encoder:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.image_projection.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_projection.parameters())

    print(f"For the Image head:")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")