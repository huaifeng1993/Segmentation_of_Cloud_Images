import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def onehot_embedding(labels,num_classes):
    N = labels.size(0)
    D = 2
    y = torch.zeros(N,D)
    y[torch.arange(0,N).long(),labels] = 1
    return y

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def focal_loss2d(input, target, start_cls_index=1,size_average=True):
    n, c, h, w = input.size()
    p = F.softmax(input)
    p = p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    p = p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= start_cls_index] #exclude background example
    p = p.view(-1, c)

    mask = target >=start_cls_index #exclude background example
    target = target[mask]

    t = onehot_embedding(target.data.cpu(),c)
    t = Variable(t).cuda()

    alpha = 0.25
    gamma = 2
    w = alpha* t + (1-alpha)*(1-t)
    w = w * (1-p).pow(gamma)

    loss = F.binary_cross_entropy(p,t,w,size_average=False)

    if size_average:
       loss /= mask.data.sum()
    return loss

def bin_clsloss(input, target, size_average=True):
    n, c = input.size()
    p = input
    target_emdding=torch.zeros((n,c))
    for i in range(n):
        nclasses = set(target.data.cpu().numpy()[i].flat)
        for nclass in nclasses:
            target_emdding[i][nclass]=1.0

    mask = target >= 0

    t = target_emdding[:,1:] #exclude background
    t = Variable(t).cuda()

    p = p[:,1:] #exclude background
    p = F.sigmoid(p) #binaray cls

    loss = F.binary_cross_entropy(p,t,size_average=size_average)

    return loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
 
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
 
        The losses are averaged across observations for each minibatch.
 
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
 
 
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        # N = inputs.size(0)
        # C = inputs.size(1)
        input=inputs.view(-1)
        P = F.sigmoid(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BCFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(BCFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            #input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.view(-1)
            #print('150:',input.size())
            #input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            #input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            #input = input.contiguous().view(-1)
            #print('154:',input.size())
        #target = target.view(-1,1)
        target = target.view(-1)
        #print('157:',target.size())

        #logpt = F.log_softmax(input)
        #logpt = F.sigmoid(input)
        #pt = F.sigmoid(input)

        logpt = F.logsigmoid(input)
        #print('162:',logpt.size())
        #logpt = logpt.gather(1,target)
        #logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        #print('166',pt.size())
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        #print('174:',loss.size())
        #print(loss)
        if self.size_average: return loss.mean()
        else: return loss.sum()



if __name__=='__main__':
    import torch.nn.functional as F
    import random
    maxe = 0
    for i in range(1000):
        p=torch.randn(1,1,56,56)*random.randint(1,10)
        p = Variable(p.cuda())
        t = torch.rand(1,1,56,56).ge(0.1).type(torch.FloatTensor)
        t = Variable(t.cuda())
        #print(p)
        #print(t) 
      
        result=BCFocalLoss()(p,t)

        result2=nn.BCEWithLogitsLoss()(p,t)
        print(result)
        print(result2)
        a=result.to('cpu').numpy()
        b=result2.to('cpu').numpy()
        if abs(a-b)>maxe: maxe = abs(a-b)
        
    print('max_error:',maxe)