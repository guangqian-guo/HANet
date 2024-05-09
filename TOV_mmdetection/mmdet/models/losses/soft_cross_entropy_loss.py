import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class SoftCrossEntropyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 bce_use_sigmoid=False,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.bce_use_sigmoid = bce_use_sigmoid

    def forward(self,
                cls_score,
                label,
                label_,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction. shape is (N ,C)
            label (torch.Tensor): The learning label of the prediction. shape is (N, C)
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        # print('calculating loss')
        # 1st version
        # cls_score = torch.sigmoid(cls_score)
        # loss = []
        # for i in range(cls_score.shape[1]):
        #     loss .append(-1 * label[:,i] * torch.log(cls_score[:,i]))
        # loss = sum(loss)
        # loss = torch.sum(loss) / avg_factor

        # 2st version
        # cls_score = torch.softmax(cls_score, dim=1)
        # loss = []
        # for i in range(cls_score.shape[1]):
        #     loss.append(-1 * torch.abs(label[:,i] - cls_score[:,i]) * label[:,i] * torch.log(cls_score[:,i]))
        # loss = sum(loss)
        # loss = torch.sum(loss) / avg_factor
        
        # 3st cersion
        # cls_score = torch.softmax(cls_score, dim=1)
        # loss = []
        # for i in range(cls_score.shape[1]):
        #     loss.append((label[:,i] - cls_score[:,i]) * label[:,i] * torch.log(label[:,i]/(cls_score[:,i]+1e-6)+1e-6))
        #     # print(label[:,i][0], cls_score[:,i][0])
        #     # print(label[:,i] * torch.log(label[:,i]/(cls_score[:,i]+1e-6)+1e-6))
        # loss = sum(loss)
        # loss = torch.sum(loss) / avg_factor
        
        # 4st version
        
        cls_score = torch.softmax(cls_score, dim=1) # N,C
        index = label_ >= 0  # 1, N, bool
        
        # for i in range(cls_score.shape[1]):
        #     loss.append((label[:,i] - cls_score[:,i]) * label[:,i] * torch.log(label[:,i]/(cls_score[:,i]+1e-6)+1e-6))
            # print(label[:,i][0], cls_score[:,i][0])
            # print(label[:,i] * torch.log(label[:,i]/(cls_score[:,i]+1e-6)+1e-6))
        # loss = (label[index,label_] - cls_score[index,label_]) * label[index,label_] * torch.log(label[index,label_]/(cls_score[index,label_]+1e-6)+1e-6)
        
        # loss = -1 * torch.log(cls_score[index,label_]+ 1e-6)   #自己设计的交叉熵损失，结果最好，没用soft的标签
        # loss = torch.sum(loss) / avg_factor
        
        # loss = -1 * label[index,label_] * torch.log(cls_score[index,label_]+ 1e-6)   # 尝试在上一个基础上，加上soft标签做权重，但是结果不好。

        #loss = torch.abs(torch.log(label[index,label_]/(cls_score[index,label_]+1e-6)+1e-6))  #怀疑是前面乘了太多小于1的值，所以去掉那些，结果损失训练时降不下去，结果不好。
        
        # loss = (label[index,label_] - cls_score[index,label_]) * torch.log(label[index,label_]/(cls_score[index,label_]+1e-6)+1e-6)

        loss = torch.abs(label[index,label_] - cls_score[index,label_]) * -1 * torch.log(cls_score[index,label_]+ 1e-6)
        loss = torch.sum(loss) / avg_factor
        # 5.1
        # cls_score = torch.sigmoid(cls_score) # N,C
        # index = label_ >= 0  # 1, N, bool
        # loss = (label[index,label_] - cls_score[index,label_]) * torch.log(label[index,label_]/(cls_score[index,label_]+1e-6)+1e-6)
        # loss = torch.sum(loss) / avg_factor
        # print(loss)
        return loss
