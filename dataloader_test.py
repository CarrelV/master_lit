import config as CFG
from dataloader import get_local_dataloader,transform_train
from dataset import get_dataset,flickr30k,mscoco

from tokenizer import get_tokenizer,get_feature_extractor


if __name__ == "__main__":

    # prepare the dataloader
    tokenizer = get_tokenizer(CFG.text_model_name)
    feature_extractor = get_feature_extractor(CFG.image_model_name)



    image_root = "data/mscoco"
    ann_root="data/mscoco/annotations"
    
    ds_train = mscoco(tokenizer=tokenizer,feature_extractor=feature_extractor,transform=transform_train,image_root=image_root,ann_root=ann_root,split="train")

    print("Train")
    for i in range(10):
        for _ in range(5):
            ds_train.__getitem__(i)
    
    