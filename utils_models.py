
from pruning import pruning_BERT_without_residual,pruning_ViT_without_residual
import config as CFG










# 
def modify_model_after_init(model,tokenizer,feature_extractor,importance_measure):

    step = CFG.ladder_reduction_factor
    if CFG.configuration == "reduced_LST_last":
        initial_gap = CFG.ladder_reduction_factor - 1
    else:
        initial_gap = 0



    if CFG.side_text_weights_copy:
        side_state_dict_text = pruning_BERT_without_residual(model.text_encoder,tokenizer,CFG.reduction_factor,importance_measure)
    
        for n,p in model.text_encoder.named_parameters():

            if "side_encoder" in n:
                
                infer_n = n.split(".")
                infer_n[0] = "encoder.layer"

                infer_n[1] = str(initial_gap + int(infer_n[1]) * step)
                infer_n = ".".join(infer_n)            

                p.data.copy_(side_state_dict_text[infer_n])

    if CFG.side_image_weights_copy:
        side_state_dict_image = pruning_ViT_without_residual(model.image_encoder,feature_extractor,CFG.reduction_factor,importance_measure)
    
        for n,p in model.image_encoder.named_parameters():

            if "side_encoder" in n:
                
                infer_n = n.split(".")
                infer_n[0] = "encoder.layer"

                infer_n[1] = str(initial_gap + int(infer_n[1]) * step)
                infer_n = ".".join(infer_n)            

                p.data.copy_(side_state_dict_image[infer_n])

    return model

def resume_model(model):

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