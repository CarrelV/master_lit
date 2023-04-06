import config as CFG
from dataloader import get_local_dataloader,transform_train
from dataset import get_dataset,flickr30k,mscoco

from tokenizer import get_tokenizer,get_feature_extractor


if __name__ == "__main__":

    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.vision_model_name)

    dataloader_train = get_local_dataloader("mscoco",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="train")
    dataloader_valid = get_local_dataloader("mscoco",tokenizer=tokenizer,feature_extractor=feature_extractor,batch_size=CFG.batch_size,shuffle=CFG.shuffle_train,split="val")

    dataset = get_dataset(dataset="mscoco",tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")

    
    
    ds_train = mscoco(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="train")

    print("Train")
    for i in range(10):

        ds_train.__getitem__(0)

    ds_val = mscoco(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,split="val")

    print("Train")
    for i in range(10):
        ds_val.__getitem__(0)