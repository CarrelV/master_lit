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


# Baseline: all random init, pretrained all
# All the other will have both backbones pretrained, and Image backbone frozen
# Baseline2: Text back frozen, head proj
# LiT: Text back finetune, head proj/small MLP 
# APE: Text back frozen, head big MLP
# APE2: Text back finetune, head big MLP ? (more than what they propose?)

configuration = "baseline"




# for projection head; used for both image and text encoders
projection_dim = 256 
dropout = 0.1

temperature = 1.0

########### MOCO Parameters ################

# length of the queue, set to 1024 to have same as batch size when frozen towers
K = 4096

m=0.999
########## Training Configuration ##########
# "classic", "moco", "ape"
model_used = "moco"
training_run_number = 1
# With frozen towers, batch size 1024 is okay
#batch_size = 1024
# When both towers are finetunes, batch size is necessary 
batch_size = 1024




num_workers = 0
shuffle_train = True
shuffle_test = False
split = "train"



# 
warming_epochs = 5
epochs = 20

image_encoder_lr = 1e-5
text_encoder_lr = 1e-4
image_head_lr = 1e-3
text_head_lr = 1e-3

weight_decay = 1e-3



########## DDP Configuration ##########

# at home = 1, cluster = 2 (or 4, check)
gpu_number = 1



########## Test Configuration ##########

checkpoint_number = 0

image_checkpoint = f"weights/img_proj_best_{checkpoint_number}.pt"
text_checkpoint = f"weights/text_proj_best_{checkpoint_number}.pt"

test_batch_size = 1

###################################################################

if configuration == "bad_baseline":
    #Model weight init
    text_backbone_pretrained = False 
    image_backbone_pretrained = False

    #Model training
    text_backbone_finetune = True 
    image_backbone_finetune = True

elif configuration == "baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

elif configuration == "costly_baseline":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = True

elif configuration == "LiT":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False

elif configuration == "APE":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = False
    image_backbone_finetune = False

elif configuration == "APE2":
    #Model weight init
    text_backbone_pretrained = True 
    image_backbone_pretrained = True

    #Model training
    text_backbone_finetune = True
    image_backbone_finetune = False
