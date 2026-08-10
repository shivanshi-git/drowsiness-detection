import torch
from .custom_cnn import CustomCNN
from .vgg_model import VGGModel
from .resnet_model import ResNetModel
from .mobilenet_model import MobileNetModel
from .efficientnet_model import EfficientNetModel
from .vit_model import ViTTinyModel

def get_model(model_name='vgg16', num_classes=2, pretrained=True):
    """
    Factory function to instantiate models by string key.
    
    Supported model_name options:
      - 'custom_cnn'
      - 'vgg16', 'vgg19'
      - 'resnet18', 'resnet50'
      - 'mobilenet_v2', 'mobilenet_v3'
      - 'efficientnet_b0', 'efficientnet_b2'
      - 'vit_tiny'
    """
    model_name = model_name.lower().strip()

    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)
        target_layer = model.features[4] # Final Residual Block (layer4)
    elif model_name in ['vgg16', 'vgg19']:
        model = VGGModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
        target_layer = model.features[-2] # Final Conv layer before pool
    elif model_name in ['resnet18', 'resnet50']:
        model = ResNetModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
        target_layer = model.backbone.layer4[-1] # Final residual block
    elif model_name in ['mobilenet_v2', 'mobilenet_v3']:
        model = MobileNetModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
        target_layer = model.backbone.features[-1]
    elif model_name in ['efficientnet_b0', 'efficientnet_b2']:
        model = EfficientNetModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
        target_layer = model.backbone.features[-1]
    elif model_name == 'vit_tiny':
        model = ViTTinyModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
        target_layer = model.backbone.encoder.layers[-1]
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from: custom_cnn, vgg16, vgg19, resnet18, resnet50, mobilenet_v2, mobilenet_v3, efficientnet_b0, vit_tiny.")

    return model, target_layer

def count_parameters(model):
    """Returns total trainable parameters count in millions."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6
