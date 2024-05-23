

#############################################################################
#                                                                           #
#                        Only stuff to Modify                               #
#                                                                           #
#############################################################################


################################
#             Training         #
################################

text_model_size = "small"
image_model_size = "small"

## See at the end the different possibilities
configuration = "baseline"
testing = False

dataset = "flickr30k"
#dataset = "mscoco"
#dataset = "cc3m"

# Increment if retraining the same configuration one more time
training_run_number = "bin_memory"

# 1024 when both backbone are frozen (baseline,good_baseline,APE)
# 64 when both backbone are finetuned (bad_baseline)
# 128 when only the text backbone is finetuned (costly_baseline,LiT,APE_LiT)
# 64 for BERT base uncased with LST
# 32 for LoRA with BERT base and ViTB16
batch_size = 32

test_batch_size = 32

# 20 / 300 on flickr
# 5 / 50 on MSCOCO
warming_epochs = 5
epochs = 50

# 1 at home, 2 on cluster
gpu_number = 2


# reduction of number of ladder connection
# default is 1, go to 2 or 4 if wanted
ladder_reduction_factor = 1

# When using LST, can chose to add a final skip connection between the output of the frozen main model and 
# the output of the upsampled side network output
sum_last_outputs = True
################################
#             Testing          #
################################

configuration_to_test = "lora"

run_info = "baseline (I=small/T=small)"
training_dataset = "mscoco"
# best or im0 or i2t or t2i
weight_information = "best"
weight_version = "lora_mscoco_Imedium6_Tsmall4"
#############################################################################
#                                                                           #
#                            END MODIFICATION                               #
#                                                                           #
#############################################################################

########## Dataset Configuration ##########

max_length = 128




prompt = ""

########## Models Configurations ##########

# Head possibilities:
# For Image: always a liner projection head
# For Text: 1) linear projection head 2) small 2 layer MLP head 3) bigger 4-6 layers MLP head 4) LST heads

# image size
size = 224

if text_model_size == "small":

    text_model_name = "prajjwal1/bert-medium"
    text_embedding = 512
    text_tower_name = "BERT medium (42M)"

elif text_model_size == "medium":

    text_model_name = "bert-base-uncased"
    text_embedding = 768

    text_tower_name = "BERT base (108M)"


if image_model_size == "small":

    image_model_name = "facebook/dino-vits16"
    image_embedding = 384

    image_tower_name = "ViT small (22M)"

elif image_model_size == "medium":

    image_model_name = "facebook/dino-vitb16"
    image_embedding = 768

    image_tower_name = "ViT base (86M)"


image_head_name = "Projection Head"
##############Pruning for LST
# reduction of hidden dim size
reduction_factor = 8


ladder_initial_gap = 0
## Side network

gate_alpha = 0.0
gate_T = 0.1


# for projection head; used for both image and text encoders
projection_dim = 256 



dropout = 0.1

temperature = 1.0

########### MOCO Parameters ################

# length of the queue, set to 1024 to have same as batch size when frozen towers
K = 4096

m=0.999

########### LoRA Parameters ################
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

# False by default, override to True if config = lora
apply_lora_text = False
apply_lora_image = False

# False by default, override to True if LST
side_text_weights_copy = False
side_image_weights_copy = False


add_final_skip_connection = False
########## Training Configuration ##########


samples_for_fisher = 128


num_workers = 8
shuffle_train = True
shuffle_test = False
split = "train"



# 

image_encoder_lr = 1e-6
text_encoder_lr = 1e-4
image_head_lr = 1e-3
text_head_lr = 1e-3

weight_decay = 1e-3





########## DDP Configuration ##########

# at home = 1, cluster = 2 (or 4, check)





########################## Different CONFIGURATION #########################################
version_add_information = "best"

if testing:
    configuration = configuration_to_test
    version_add_information = weight_information

if configuration == "lora_text":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = False

    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


    apply_lora_text = True

elif configuration == "lora_image":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False 
    image_backbone_finetune = True

    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


    apply_lora_image = True

elif configuration == "lora":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = True

    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


    apply_lora_text = True
    apply_lora_image = True


elif configuration == "text_LST":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = False
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "LST"
    image_tower_config = "classic"
    find_unused_param = False

    side_text_weights_copy = True
    
    add_final_skip_connection = True


elif configuration == "image_LST":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False 
    image_backbone_finetune = True
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "LST"
    find_unused_param = False

    side_image_weights_copy = True
    
    add_final_skip_connection = True

elif configuration == "LST":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = True
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "LST"
    image_tower_config = "LST"
    find_unused_param = False

    side_image_weights_copy = True
    side_text_weights_copy = True
    
    add_final_skip_connection = True

elif configuration == "reduced_LST_first":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = False
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "LST"
    image_tower_config = "classic"
    find_unused_param = False

    side_text_weights_copy = True
    
    add_final_skip_connection = True

    ladder_reduction_factor = 4

elif configuration == "reduced_LST_last":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = False
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "LST"
    image_tower_config = "classic"
    find_unused_param = False

    side_text_weights_copy = True
    
    add_final_skip_connection = True

    ladder_reduction_factor = 4
    ladder_initial_gap = ladder_reduction_factor - 1

elif configuration == "baseline_transformer":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "transformer_head"
    text_head_name = "Transformer Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


elif configuration == "bad_baseline":
    #Model weight init
    text_backbone_pretrained = False 
    image_backbone_pretrained = False

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = True
    
    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


elif configuration == "baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False



elif configuration == "good_baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "small_mlp"
    text_head_name = "Small MLP Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


elif configuration == "classic_LiT":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

    text_head_config = "simple_proj"
    text_head_name = "Projection Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


elif configuration == "LiT":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

    text_head_config = "small_mlp"
    text_head_name = "Small MLP Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


elif configuration == "APE":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "large_mlp"
    text_head_name = "Medium MLP Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


#Using both the LiT finetuning scheme and the bigger APE MLP
elif configuration == "APE_LiT":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

    text_head_config = "large_mlp"
    text_head_name = "Medium MLP Head"
    text_tower_config = "classic"
    image_tower_config = "classic"
    find_unused_param = False


