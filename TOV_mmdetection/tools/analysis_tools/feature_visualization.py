import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

from mmdet.models import (CenterNet_Decouple)


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(img, features,save_dir = 'feature_map',name = None):
    # img = mmcv.imread(img_file)
    # model.eval()
    # features = inference_detector(model, img)
    print(img.shape)
    img = img.squeeze().permute(2,1,0)
    img = img.detach().cpu().numpy()
    # img = cv2.cvtColor(img,cv2.COLORMAP_JET)
    cv2.imshow('img',img)
    cv2.waitKey(500)
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (640, 512))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)

            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                # print(heatmap.shape)
                # assert 1==0
                heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                cv2.imshow("1",superimposed_img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir+str(i)+'.png'), superimposed_img)
                i=i+1

def draw_feature_map_(imgfile, model,save_dir):
    i = 0
    img = mmcv.imread(imgfile)
    features = inference_detector(model, img)
    print(len(features))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if isinstance(features, torch.Tensor):
        for heat_maps in features:
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (640, 512))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                cv2.imshow("1", superimposed_img)
                cv2.waitKey(50)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir + str(i) + '.png'), superimposed_img)
                i = i + 1


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='demo1.jpg', help='Image file')
    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint-file',help='checkpoint file')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    return args



def main(args):
    model = init_detector(args.config, args.checkpoint_file, args.device)

    draw_feature_map_(args.img, model, save_dir='feature_visiualization_conv/')


if __name__ == '__main__':
    args = parse_args()
    main(args)
