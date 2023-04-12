import os
import json


if __name__ == "__main__":
    
    with open('imageNet_utils/imagenet_class_index.json') as f:
        labels = json.load(f)

    for i in labels.values():
        os.mkdir(f"ImageNet/train/{i[0]}")