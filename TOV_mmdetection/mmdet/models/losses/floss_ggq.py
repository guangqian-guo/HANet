import torch
from ..builder import LOSSES
@LOSSES.register_module()
class Focal_Loss_G():
    def __init__(self,weight=1,gamma=2):
        super(Focal_Loss_G,self).__init__()
        self.gamma=gamma
        self.weight=weight

    def forward(self,preds,labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps=1e-7
        y_pred =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
        
        target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
        
        ce=-1*torch.log(y_pred+eps)*target
        floss=torch.pow((1-y_pred),self.gamma)*ce
        floss=torch.mul(floss,self.weight)
        floss=torch.sum(floss,dim=1)
        return torch.mean(floss)
