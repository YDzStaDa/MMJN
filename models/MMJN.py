import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from torch.nn.parameter import Parameter

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        out = torch.bmm(attention, proj_value).view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class CNNS1(Model):  # test
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        self.classnames=[]
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            self.net_2._modules['6'] = nn.Linear(4096,40)
    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0],-1))

class CNNS2(Model):  # test
    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=4):
        self.classnames=[]
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))

class CNNS3(Model):
    def __init__(self, name, sv_model, mv_model, nclasses=40, num_views=5):
        self.nclasses = nclasses
        self.num_views = num_views
        self.sv = sv_model
        self.mv = mv_model
        self.attention = SelfAttention(in_dim=512)
        self.svcnn_weight = Parameter(torch.tensor(0.63))
    def forward(self, x):
        svoutput = self.sv.net_1(x[:, 0, :, :, :])
        mx = x[:, 1:, :, :, :]
        mvinput = mx.contiguous().view(-1, mx.shape[-3], mx.shape[-2], mx.shape[-1])
        mvoutput = self.mv.net_1(mvinput)
        mvoutput = mvoutput.view((-1, self.num_views - 1, mvoutput.shape[-3], mvoutput.shape[-2], mvoutput.shape[-1]))
        mvoutput = torch.max(mvoutput,1)[0].view(mvoutput.shape[0], mvoutput.shape[-3], mvoutput.shape[-2], mvoutput.shape[-1])
        cooutput = self.svcnn_weight * svoutput + (1 - self.svcnn_weight) * mvoutput
        cooutput = self.attention(cooutput)
        cooutput = cooutput.view(cooutput.shape[0], -1)
        return self.mvcnn.net_2(cooutput)





