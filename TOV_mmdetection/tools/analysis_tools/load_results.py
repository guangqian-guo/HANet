import json
from matplotlib import image

from matplotlib.pyplot import annotate
from PIL import Image,ImageDraw
import torch
import numpy as np
results_file = '/home/ubuntu/Guo/TOV_mmdetection-main/work-dir/centernet-r50/results/val_results.bbox.json'
anno_file = '/home/ubuntu/Guo/TOV_mmdetection-main/data/tiny_set/annotations/task/tiny_set_test_all.json'
img_root = './data/tiny_set/test/'

img_idx = {}
def nms( bboxes, scores, threshold=0.5):
    print(bboxes[0])
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2] + bboxes[:,0]
    y2 = bboxes[:,3] + bboxes[:,1]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return keep   # Pytorch的索引值为LongTensor


with open(anno_file,encoding='utf-8') as f:
    d = json.load(f)
    images = d["images"]
    for i in images:
        id = i["id"]
        img_idx[id] = i["file_name"]
# print(img_idx)
f.close()


img_id = 794
bboxes = []
scores = []
with open(results_file, encoding='utf-8') as f:
    d = json.load(f)

    for i in d:
        if img_id == i["image_id"]:
            img_file = img_idx[i["image_id"]]
            img_path = img_root + img_file
            bbox = i["bbox"]
            bboxes.append(bbox)
            scores.append(i["score"])
        else:
            break
    print(len(bboxes))
    # img = Image.open(img_path)
    # draw = ImageDraw.Draw(img)
    after_nms = nms(torch.tensor(bboxes), torch.tensor(scores))
    print(len(after_nms))

