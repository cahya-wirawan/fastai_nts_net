from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.padding import ZeroPad2d

from fastai.torch_core import *
from fastai.basic_train import *
from fastai.basic_data import *
from fastai.layers import *
from fastai.callback import *
from fastai.layers import *
from fastai.callbacks.hooks import *
from fastai.vision.learner import *

drivedownloader = try_import('google_drive_downloader')
if not drivedownloader:
    raise Exception('Error: `googledrivedownloader` is needed. `pip install googledrivedownloader`')
from google_drive_downloader import GoogleDriveDownloader as gdd

# Most of the following code is from imgclsmob's repository, as their model I was able to get working via split()
                    
from .resnet import *
from .anchors import generate_default_anchor_maps, hard_nms


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class NTSNet(nn.Module):
    def __init__(self, data:DataBunch, backbone:nn.Sequential, topN=6,  cat_num:int=4):
        super(NTSNet, self).__init__()
        self.cat_num=cat_num
        self.classes = data.c
        size = data.train_ds.tfmargs.get('size')
        self.in_size = (size, size)
        self.size = size
        self.size_t = size//2
                 
        self.backbone = backbone
        self.backbone_tail = nn.Sequential()
        self.backbone_tail.add_module('Final Pool', nn.AdaptiveAvgPool2d(1))
        self.backbone_tail.add_module('Flatten', Flatten())
        self.backbone_tail.add_module('Dropout', nn.Dropout(p=0.5))
                 
        self.backbone_classifier = nn.Linear(512*4, data.c)
        
        
        self.topN = topN
                 
        _, edge_anchors, _ = generate_default_anchor_maps(input_shape=self.in_size)
        self.pad_side = size//2
                 
        self.edge_anchors = (edge_anchors + self.pad_side).astype(np.int)         
        self.edge_anchors = np.concatenate(
            (self.edge_anchors.copy(), np.arange(0, len(self.edge_anchors)).reshape(-1,1)), axis=1)
                 
        self.pad = ZeroPad2d(padding=self.size_t)
                               
        self.proposal_net = ProposalNet()
        # self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 200)
        self.concat_net = nn.Linear(2048 * (self.cat_num + 1), data.c)
        # self.partcls_net = nn.Linear(512 * 4, 200)
        self.partcls_net = nn.Linear(512 * 4, data.c)
        

    def forward(self, x):
        
        raw_pre = self.backbone(x)         
        rpn_score = self.proposal_net(raw_pre)
                 
                 
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        #rpn_score = self.proposal_net(rpn_feature.detach())
                 
        all_cdds = [np.concatenate((y.reshape(-1, 1), self.edge_anchors.copy()), axis=1)
                    for y in rpn_score.detach().cpu().numpy()]
                 
        top_n_cdds = [hard_nms(y, topn=self.topN, iou_thresh=0.25) for y in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64) # when running code, here went a error, change np.int to np.int64,parameter index of torch.gather() appoint longtensortype when change num_worker to 4,then it runs on windows or linux correctly
        top_n_index = torch.from_numpy(top_n_index).long().to(x.device)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
                 
        part_imgs = torch.zeros([batch, self.topN, 3, self.size_t, self.size_t]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(self.size_t, self.size_t), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, self.size_t, self.size_t)
                 
        part_features = self.backbone_tail(self.backbone(part_imgs.detach()))
                 
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :self.cat_num, :].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        raw_features = self.backbone_tail(raw_pre.detach())
                 
        concat_out = torch.cat([part_feature, raw_features], dim=1)
        concat_logits = self.concat_net(concat_out)
                 
        raw_logits = self.backbone_classifier(raw_features)
       
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return concat_logits, raw_logits, part_logits, top_n_prob

def _nts_split(m:nn.Sequential) -> List[nn.Module]:
    groups = [[*list(m.children())[:1]]]
    groups += [[*list(m.children())[1:]]]
    return groups



def get_head(data:DataBunch, pretrained=False):
    path=data.path
    net = NTSNet(data, pretrained)
    if pretrained:
        model_dict = net.state_dict()
        pre_dict = torch.load(Path(path/'Pretrained-Weights.pth'))['model']

        #for name, param in model_dict.items():
        #    if name in pre_dict:
        #        input_param = pre_dict[name]
        #        if input_param.shape == param.shape:
        #            param.copy_(input_param)

        pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pre_dict) 
        
        net.load_state_dict(model_dict)
        
    
    h1 = [*list(net.children())[1:]]
        
    head = nn.Sequential(*list(h1))
    return head
