from models.segmentation._deeplab.modeling import deeplabv3_hrnetv2_48


class DeepLabV3HRNetV2W48:

    def __new__(cls, config, *args, **kwargs):
        model = deeplabv3_hrnetv2_48(num_classes=config['n_classes'], pretrained_backbone=False)
        model.n_classes = config['n_classes']
        return model
