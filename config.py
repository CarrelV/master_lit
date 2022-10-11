import torch

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
trainable = False # for both image encoder and text encoder

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1

temperature = 1.0


########## Training Configuration ##########

batch_size = 1024
num_workers = 0
shuffle = False
split = "train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 2

image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
head_lr = 1e-3
weight_decay = 1e-3

#LR scheduler
patience = 2
factor = 0.5
