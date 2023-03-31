
from pruning import pruning_BERT_without_residual
import config as CFG










# 
def modify_text_model_after_init(model,tokenizer):

    side_state_dict = pruning_BERT_without_residual(model.text_encoder,tokenizer,CFG.reduction_factor)

    for n,p in model.text_encoder.named_parameters():

        if "side_encoder" in n:

            infer_n = n.split(".")
            infer_n[0] = "encoder.layer"
            infer_n = ".".join(infer_n)            

            p.data.copy_(side_state_dict[infer_n])


    return model