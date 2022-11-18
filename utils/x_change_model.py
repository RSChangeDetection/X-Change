from utils.res_unet import *
from backbone.resnet import *


class Resnet_CD(nn.Module):
    def __init__(self, backbone='18', x_cross=False, operation_type='cat'):
        super(Resnet_CD, self).__init__()
        if backbone == '18':
            self.backbone = resnet18()
            self.change_head = res_unet(n_channels=3, n_classes=1, x_cross=x_cross, op_type=operation_type, factor=1)
        if backbone == '34':
            self.backbone = resnet34()
            self.change_head = res_unet(n_channels=3, n_classes=1, x_cross=x_cross, op_type=operation_type, factor=1)
        if backbone == '50':
            self.backbone = resnet50()
            self.change_head = res_unet(n_channels=3, n_classes=1, x_cross=x_cross, op_type=operation_type, factor=4)
        if backbone == '101':
            self.backbone = resnet101()
            self.change_head = res_unet(n_channels=3, n_classes=1, x_cross=x_cross, op_type=operation_type, factor=4)

    def forward(self, before, after):
        befores = self.backbone(before)
        afters = self.backbone(after)
        pred = self.change_head(befores, afters)
        return pred
