import torch

from models.segmentation._deeplab.modeling import deeplabv3_mobilenet


class DeepLabV3MobileNet:

    def __new__(cls, config, *args, **kwargs):
        model = deeplabv3_mobilenet(num_classes=config['n_classes'], pretrained_backbone=False)
        model.n_classes = config['n_classes']
        return model
