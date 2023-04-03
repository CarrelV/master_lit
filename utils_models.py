
from pruning import pruning_BERT_without_residual
import config as CFG










# 
def modify_text_model_after_init(model,tokenizer,importance_measure,device):

    side_state_dict = pruning_BERT_without_residual(model.text_encoder,tokenizer,CFG.reduction_factor,importance_measure)

    for n,p in model.text_encoder.named_parameters():

        if "side_encoder" in n:

            infer_n = n.split(".")
            infer_n[0] = "encoder.layer"
            infer_n = ".".join(infer_n)            

            p.data.copy_(side_state_dict[infer_n])

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