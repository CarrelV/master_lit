########## Dataset Configuration ##########

image_root=""
ann_root="./flickr30k"

max_length = 128
prompt = ""

########## Models Configurations ##########

# Head possibilities:
# For Image: always a liner projection head
# For Text: 1) linear projection head 2) small 2 layer MLP head 3) bigger 4-6 layers MLP head 4) LST heads

# image size
size = 224

text_model_name = "prajjwal1/bert-medium"
text_embedding = 512

vision_model_name = "facebook/dino-vits16"
image_embedding = 384



#############################################################################
#                                                                           #
#                        Only stuff to Modify                               #
#                                                                           #
#############################################################################


################################
#             Training         #
################################

## See at the end the different possibilities
configuration = "APE_LiT"
testing = True

# Increment if retraining the same configuration one more time
training_run_number = 1

# 1024 when both backbone are frozen (baseline,good_baseline,APE)
# 64 when both backbone are finetuned (bad_baseline)
# 128 when only the text backbone is finetuned (costly_baseline,LiT,APE_LiT)
batch_size = 128

warming_epochs = 20
epochs = 300

# 1 at home, 2 on cluster
gpu_number = 2


################################
#             Testing          #
################################

configuration_to_test = "LiT"

weight_version = 1
#############################################################################
#                                                                           #
#                            END MODIFICATION                               #
#                                                                           #
#############################################################################


# for projection head; used for both image and text encoders
projection_dim = 256 
dropout = 0.1

temperature = 1.0

########### MOCO Parameters ################

# length of the queue, set to 1024 to have same as batch size when frozen towers
K = 4096

m=0.999
########## Training Configuration ##########





num_workers = 0
shuffle_train = True
shuffle_test = False
split = "train"



# 

image_encoder_lr = 1e-5
text_encoder_lr = 1e-4
image_head_lr = 1e-3
text_head_lr = 1e-3

weight_decay = 1e-3



########## DDP Configuration ##########

# at home = 1, cluster = 2 (or 4, check)





########################## Different CONFIGURATION #########################################

if testing:
    configuration = configuration_to_test


if configuration == "bad_baseline":
    #Model weight init
    text_backbone_pretrained = False 
    image_backbone_pretrained = False

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = True
    
    text_head_config = "simple_proj"
    find_unused_param = True

elif configuration == "baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "simple_proj"
    find_unused_param = False


elif configuration == "good_baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "small_mlp"
    find_unused_param = False

elif configuration == "costly_baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

    text_head_config = "simple_proj"
    find_unused_param = True

elif configuration == "LiT":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

    text_head_config = "small_mlp"
    find_unused_param = True

elif configuration == "APE":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

    text_head_config = "large_mlp"
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
    find_unused_param = True
