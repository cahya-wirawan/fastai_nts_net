from fastai import *
from fastai.vision.learner import _resnet_split
from model import *
from prediction import *
from loss_functions import *
from model import _nts_split


def ntsnet_learner(data, original_resnet=False):

    body = create_body(models.resnet50, pretrained=True)
    init = nn.init.kaiming_normal_
    if not original_resnet:
        net = NTSNet(data, body, 6, 4)
        learn = Learner(data, net, loss_func=total_loss, metrics=metric)
        learn.split(_nts_split)
        learn.freeze()

        apply_init(learn.model.backbone_classifier, init)
        apply_init(learn.model.backbone_tail, init)
        apply_init(learn.model.proposal_net, init)
        apply_init(learn.model.partcls_net, init)
        apply_init(learn.model.concat_net, init)
        apply_init(learn.model.pad, init)
    else:
        nf = num_features_model(nn.Sequential(*body.children())) * 2
        head = create_head(nf, data.c)
        model = nn.Sequential(body, head)
        learn = Learner(data, model, loss_func=total_loss, metrics=metric)
        learn.split(_resnet_split)
        learn.freeze()
        apply_init(learn.model[1], init)

    return learn
