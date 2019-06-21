from fastai import *
from fastai.vision.learner import _resnet_split
from .model import *
from .loss_functions import *
from .model import _nts_split


def freeze_layer(model):
    """
    This function freeze the layer and its children, except any batch normalisation.
    """
    for node in list(model.children()):
        children = list(node.children())
        if len(children) == 0:
            if isinstance(node, bn_types):
                requires_grad(node, True)
            else:
                requires_grad(node, False)
        else:
            freeze_layer(node)

def ntsnet_learner(data, original_resnet=False, loss_func=total_loss, metrics=metric,**kwargs:Any):

    body = create_body(models.resnet50, pretrained=True)
    init = nn.init.kaiming_normal_
    if not original_resnet:
        net = NTSNet(data, body, 6, 4)
        learn = Learner(data, net, loss_func=loss_func, metrics=metrics, **kwargs)
        learn.split(_nts_split)
        learn.freeze()
        freeze_layer(learn.model.backbone)
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
        learn = Learner(data, model, metrics=accuracy)
        learn.split(_resnet_split)
        learn.freeze()
        apply_init(learn.model[1], init)

    return learn
