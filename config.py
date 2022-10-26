########## Dataset Configuration ##########

image_root=""
ann_root="./flickr30k"

max_length = 128
prompt = ""

########## Models Configurations ##########

# image size
size = 224

text_model_name = "prajjwal1/bert-medium"
text_embedding = 512

vision_model_name = "facebook/dino-vits16"
image_embedding = 384

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1

temperature = 1.0

########### MOCO Parameters ################

# length of the queue, set to 1024 to have same as batch size when frozen towers
K = 1024

m=0.999
########## Training Configuration ##########
# "classic", "moco", "ape"
model_used = "moco"
training_run_number = 1
# With frozen towers, batch size 1024 is okay
#batch_size = 1024
# When both towers are finetunes, batch size is necessary 
batch_size = 64




num_workers = 0
shuffle_train = True
shuffle_test = False
split = "train"



# 
epochs = 500

image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
head_lr = 1e-3
weight_decay = 1e-3

#LR scheduler
T_max = 10


########## DDP Configuration ##########

# at home = 1, cluster = 2 (or 4, check)
gpu_number = 1



########## Test Configuration ##########

checkpoint_number = 0

image_checkpoint = f"weights/img_proj_best_{checkpoint_number}.pt"
text_checkpoint = f"weights/text_proj_best_{checkpoint_number}.pt"

test_batch_size = 1