import os
import torch
import torch.nn as nn
from torchvision.models import resnet50

def main():
    device = torch.device('cuda:0')

    model_weight_path = '/home/ubuntu/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'


    net = resnet50(num_classes=5)
    pre_weight = torch.load(model_weight_path, map_location=device)
    print(pre_weight.keys())




if __name__ == '__main__':
    main()
