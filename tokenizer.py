from transformers import BertTokenizerFast
from transformers import ViTFeatureExtractor

def get_tokenizer(text_model_name):

    return BertTokenizerFast.from_pretrained(text_model_name)

def get_feature_extractor(image_model_name):

    return ViTFeatureExtractor.from_pretrained(image_model_name)

def encode_batch(batch,tokenizer):
    encoded_text = tokenizer(batch["caption"],padding=True,return_tensors="pt")
    #encoded_image = feature_extractor(batch["image"])
    item = {"image" : batch["image"],"input_ids": encoded_text["input_ids"],"attention_mask":encoded_text["attention_mask"]}

    return item