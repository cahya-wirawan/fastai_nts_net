from .model import *
from .loss_functions import *
from .model import _nts_split


def ntsnet_learner(data, loss_func=total_loss, metrics=metric,**kwargs:Any):
    backbone = models.resnet50
    net = NTSNet(data, backbone, 6, 4)
    learn = Learner(data, net, loss_func=loss_func, metrics=metrics, **kwargs)
    learn.split(_nts_split)
    learn.freeze()
    #freeze_layer(learn.model.backbone)
    #apply_init(learn.model.backbone_tail, init)
    #apply_init(learn.model.backbone_classifier, init)
    #apply_init(learn.model.proposal_net, init)
    #apply_init(learn.model.partcls_net, init)
    #apply_init(learn.model.concat_net, init)
    #apply_init(learn.model.pad, init)

    return learn
