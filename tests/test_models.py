import unittest
import torch
from models.backbones.resnet50 import ResNet50Baseline
from models.backbones.inceptionv3 import InceptionV3Baseline
from models.backbones.vit_baseline import ViTBaseline
from models.backbones.swin_baseline import SwinTransformerBaseline
from models.llformer import LLFormer
from models.region_vit import RegionAwareViT
from models.flow_vit import OpticalFlowViT


class TestModelArchitectures(unittest.TestCase):
    def test_backbones(self):
        x = torch.rand(2, 3, 224, 224)
        resnet = ResNet50Baseline(num_classes=5, pretrained=False)
        self.assertEqual(resnet(x)["logits"].shape, (2, 5))

        vit = ViTBaseline(num_classes=5, pretrained=False)
        self.assertEqual(vit(x)["logits"].shape, (2, 5))

        swin = SwinTransformerBaseline(num_classes=5, pretrained=False)
        self.assertEqual(swin(x)["logits"].shape, (2, 5))

    def test_custom_transformers(self):
        llformer = LLFormer(dim=16, num_blocks=1)
        x = torch.rand(2, 3, 112, 112)
        self.assertEqual(llformer(x).shape, (2, 3, 112, 112))

        region_vit = RegionAwareViT(embed_dim=128, depth=2, num_heads=4)
        f = torch.rand(2, 3, 224, 224)
        le = torch.rand(2, 3, 64, 64)
        re = torch.rand(2, 3, 64, 64)
        m = torch.rand(2, 3, 64, 64)
        self.assertEqual(region_vit(f, le, re, m).shape, (2, 245, 128))


if __name__ == "__main__":
    unittest.main()
