from models.segmentation._deeplab.modeling import deeplabv3_resnet50


class DeepLabV3ResNet50:

    def __new__(cls, config, *args, **kwargs):
        model = deeplabv3_resnet50(num_classes=config['n_classes'], pretrained_backbone=False)
        model.n_classes = config['n_classes']
        return model
