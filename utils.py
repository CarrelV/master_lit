import torch.distributed as dist
import os
import torch
from pathlib import Path
import json
from collections import OrderedDict

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

'''
# in case we load a DDP model checkpoint to a non-DDP model
model_dict = OrderedDict()
pattern = re.compile('module.')
for k,v in state_dict.items():
    if re.search("module", k):
        model_dict[re.sub(pattern, '', k)] = v
    else:
        model_dict = state_dict
model.load_state_dict(model_dict)'''


def read_imagenet_class():
    classes = []
    with open(r"imagenet_classes.txt", 'r') as fp:
        for line in fp:
        
            x = line[:-1]

            classes.append(x)
    return classes

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)