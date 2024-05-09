from audioop import avg
from distutils.command.build import build
from math import ceil
from turtle import width
from numpy import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head_decouple import BaseDenseHeadDec
from .dense_test_mixins import BBoxTestMixin
from external_attention_pytorch.model.attention.CBAM import CBAMBlock 
import numpy as np
import cv2
@HEADS.register_module()
class CenterNetMSHeadAFSoftAssignment2(BaseDenseHeadDec, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_mosaic = dict(type='SoftCrossEntropyLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super(CenterNetMSHeadAFSoftAssignment2, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        #
        self.loss_mosaic = build_loss(loss_mosaic)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(feat_channel, out_channel, kernel_size=5, padding=2))
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        # print(feat.shape)
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        # center_heatmap_pred = self.heatmap_head(feat)
        
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred
        
    

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             mosaic_maps,
             x,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        # print(len(mosaic_maps), len(x))
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) 
        loss_center_heatmap = torch.tensor(0,device='cuda').float()
        loss_wh = torch.tensor(0,device='cuda').float()
        loss_offset = torch.tensor(0,device='cuda').float()
        loss_mosaic = torch.tensor(0, device='cuda').float()
        #====================== added by guo ==============================================================#
        self.area_range = self.train_cfg.area_range
        loss_factor = torch.ones(len(center_heatmap_preds))

        #====================== added by guo ==============================================================#
        for i in range(len(center_heatmap_preds)):
            target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                         center_heatmap_preds[i].shape,
                                                         img_metas[0]['pad_shape'], area_range=self.area_range[i])
            # print(avg_factor)
            center_heatmap_target = target_result['center_heatmap_target']

            wh_target = target_result['wh_target']
            offset_target = target_result['offset_target']
            wh_offset_target_weight = target_result['wh_offset_target_weight']
            mosaic_target = target_result['mosaic_target']
            mosaic_target_ = target_result['mosaic_target_']
            # Since the channel of wh_target and offset_target is 2, the avg_factor
            # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        #---------------------------------------------------------------------------------------------------------------#
            # loss_center_heatmap.append(self.loss_center_heatmap(
            #      center_heatmap_preds[i], center_heatmap_target, avg_factor=avg_factor) * loss_factor[i])
            #
            #
            # loss_wh.append(self.loss_wh(
            #     wh_preds[i],
            #     wh_target,
            #     wh_offset_target_weight,
            #     avg_factor=avg_factor * 2) * loss_factor[i])
            #
            #
            # loss_offset.append(self.loss_offset(
            #     offset_preds[i],
            #     offset_target,
            #     wh_offset_target_weight,
            #     avg_factor=avg_factor * 2) * loss_factor[i])
        #---------------------------------------------------------------------------------------------------------------#

            loss_center_heatmap += (self.loss_center_heatmap(
                center_heatmap_preds[i], center_heatmap_target, avg_factor=avg_factor) * loss_factor[i])
            # print(avg_factor)
            loss_wh += (self.loss_wh(
                wh_preds[i],
                wh_target,
                wh_offset_target_weight,
                avg_factor=avg_factor * 2) * loss_factor[i])


            loss_offset += (self.loss_offset(
                offset_preds[i],
                offset_target,
                wh_offset_target_weight,
                avg_factor=avg_factor *2 ) * loss_factor[i])
            
            #
            if mosaic_maps is not None:
                mosaic_target = mosaic_target.permute(0,2,3,1).reshape(-1,5)
                mosaic_target_ = mosaic_target_.reshape(-1)
                mosaic_map = mosaic_maps[i].permute(0, 2, 3,
                                      1).reshape(-1, 5)
                loss_mosaic += (self.loss_mosaic(mosaic_map, mosaic_target.long(),mosaic_target_.long(), avg_factor=mosaic_map.shape[0]
                                ))
            else:
                loss_mosaic += torch.tensor(0, device='cuda').float()
        #--------------------------------------------------------------------------------------------------------------#
        # print(loss_center_heatmap, loss_offset, loss_wh)
        # return dict(
        #     loss_center_heatmap0=loss_center_heatmap[0],
        #     loss_center_heatmap1=loss_center_heatmap[1],
        #     loss_center_heatmap2=loss_center_heatmap[2],
        #     loss_center_heatmap3=loss_center_heatmap[3],
        #     loss_center_heatmap4=loss_center_heatmap[4],
        #     loss_wh0=loss_wh[0],
        #     loss_wh1=loss_wh[1],
        #     loss_wh2=loss_wh[2],
        #     loss_wh3=loss_wh[3],
        #     loss_wh4=loss_wh[4],
        #     loss_offset0=loss_offset[0],
        #     loss_offset1=loss_offset[1],
        #     loss_offset2=loss_offset[2],
        #     loss_offset3=loss_offset[3],
        #     loss_offset4=loss_offset[4]
        #     )
        #---------------------------------------------------------------------------------------------------------------#
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_mosaic = loss_mosaic)

    def get_targets(self, gt_bboxes, gt_labels,feat_shape, img_shape,area_range):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        assert width_ratio == height_ratio
        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])
        mosaic_target = gt_bboxes[-1].new_zeros([bs, 5, int(feat_h/4), int(feat_w/4)])
        mosaic_target_ = gt_bboxes[-1].new_zeros([bs, 1, int(feat_h/4), int(feat_w/4)])
        
        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]

            mosaic_target[batch_id], mosaic_target_[batch_id] = self.get_mosaic_target(gt_bbox, width_ratio, feat_h, feat_w, self.area_range)
            
            area = torch.sqrt((gt_bbox[:,[2]] - gt_bbox[:,[0]]) * (gt_bbox[:, [3]] - gt_bbox[:, [1]]))
            condition = (area>area_range[0]) & (area < area_range[1])
            # print(condition)
            idx = torch.where(condition)
            gt_bbox = gt_bbox[idx[0]]
            gt_label = gt_label[idx[0]]
            try:
                center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
                center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
                gt_centers = torch.cat((center_x, center_y), dim=1)
            except:
                gt_centers = None

            for j, ct in enumerate(gt_centers):
                ctx, cty = ct
                if ctx > feat_w:
                    print(feat_w)
                    print("warning: ctx={%f} is out of range"%ctx)
                    ctx = feat_w-1
                if cty > feat_h:
                    print(feat_h)
                    print("warning: cty={%f} is out of range" % cty)
                    cty = feat_h - 1
                ctx_int, cty_int = int(ctx), int(cty)
            #-------------------------------------------- changed by guo ---------------------------------------#
                # ctx, cty = ct
                # if torch.round(ctx) >= feat_w:
                #     print(feat_w)
                #     print("warning: ctx={%f} is out of range" % ctx)
                #     ctx = torch.tensor(feat_w - 1, device='cuda').float()
                # if torch.round(cty) >= feat_h:
                #     print(feat_h)
                #     print("warning: cty={%f} is out of range" % cty)
                #     cty = torch.tensor(feat_h - 1, device='cuda').float()
                # ctx_int, cty_int = int(torch.round(ctx)), int(torch.round(cty))
            #-----------------------------------------------------------------------------------------------#
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                bbox_size = (gt_bbox[j][3] - gt_bbox[j][1]) * (gt_bbox[j][2] - gt_bbox[j][0])
                
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                # print(radius,int(radius),bbox_size)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                try:
                    wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                    wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                except:
                    print('cty:', cty_int, 'ctx:', ctx_int)
                    print('wh_target:', wh_target.shape)
                    print('img_h:', img_h, 'img_w:', img_w)
                    print('feat_h:', feat_h, 'feat_w', feat_w)
                    print('gt_box', gt_bbox)
                    print('width_ratio:', width_ratio, 'height_ratio:', height_ratio)
                    print('gt_centers:', gt_centers)
                    assert 1 == 0
                
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1


        torch.set_printoptions(profile='full')
        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
            mosaic_target=mosaic_target,
            mosaic_target_=mosaic_target_)
        return target_result, avg_factor

    # added by guo
    def get_mosaic_target(self, gt_bbox, width_ratio, feat_h, feat_w, area_ranges):
        
        mask = np.zeros([int(feat_h/4), int(feat_w/4)])
        mask_score = np.zeros([5, int(feat_h/4), int(feat_w/4)]) # h w 5
        mask_score[0,:,:] = 1      #BG=1

        area = torch.sqrt((gt_bbox[:,[2]] - gt_bbox[:,[0]]) * (gt_bbox[:, [3]] - gt_bbox[:, [1]]))
        for id, area_range in enumerate(area_ranges):
            condition = (area>area_range[0]) & (area < area_range[1])
            idx = torch.where(condition)
            gt_bbox_scale = gt_bbox[idx[0]]
            gt_bbox_scale = gt_bbox_scale * width_ratio / 4
            area_scale = area[idx[0]]
            max_range = (area_ranges[-2][1] - area_ranges[-2][0])

            for box, area_ in zip(gt_bbox_scale, area_scale):
                box, area_ = box.cpu(), area_.cpu()[0]
                rect = np.array([[box[0],box[1]], [box[2],box[1]], [box[2],box[3]], [box[0],box[3]]], np.int32)
                cv2.fillConvexPoly(mask_score[0], rect, 0)
                if area_ < ((area_range[1] - area_range[0]) / 2) + area_range[0] and id > 0:
                    dieta = (max_range / (area_range[1] - area_range[0])) * np.abs(area_ - area_range[0])
                    score = soft_label(np.array(dieta))
                    cv2.fillConvexPoly(mask_score[id+1], rect, score)
                elif id < 3:
                    dieta = (max_range / (area_range[1] - area_range[0])) * np.abs(area_range[1] - area_)
                    score = soft_label(np.array(dieta))
                    cv2.fillConvexPoly(mask_score[id+1], rect, score)
                cv2.fillConvexPoly(mask, rect, id+1)

        return torch.tensor(mask_score, device='cuda'),torch.tensor(mask, device='cuda')
    
    
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds)
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        batch_det_bboxes_multi_layers = []
        batch_labels_multi_layers = []
        for i in range(len(center_heatmap_preds)):

            batch_det_bboxes, batch_labels = self.decode_heatmap(
                center_heatmap_preds[i],
                wh_preds[i],
                offset_preds[i],
                img_metas[0]['batch_input_shape'],
                k=self.test_cfg.topk,
                kernel=self.test_cfg.local_maximum_kernel)
            batch_det_bboxes_multi_layers.append(batch_det_bboxes)
            batch_labels_multi_layers.append(batch_labels)

        batch_det_bboxes = torch.cat(batch_det_bboxes_multi_layers, dim=1)
        batch_labels = torch.cat(batch_labels_multi_layers, dim=1)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results
    

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        # topk_xs = topk_xs + 3
        # topk_ys = topk_ys + 3
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

def soft_label(x):
    z = np.exp(-x)
    sig = 1 / (1+z)
    return (sig)


