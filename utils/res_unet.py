""" Full assembly of the parts to form the complete network """
from utils.utils import *
from cross_attention.x_cross import Cross_Attention


class res_unet(nn.Module):
    def __init__(self, n_channels, n_classes, x_cross=False, bilinear=False, op_type='cat', factor=1):
        super(res_unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.x_cross = x_cross

        self.up1 = Up(512 * factor, 256 * factor, bilinear)
        self.up2 = Up(256 * factor, 128 * factor, bilinear)
        self.up3 = Up(128 * factor, 64 * factor, bilinear)

        self.c_up2 = Up(256 * factor, 128 * factor, bilinear, change_agg=True)
        self.c_up3 = Up(128 * factor, 64 * factor, bilinear, change_agg=True)
        self.c_up4 = Up(64 * factor, 32 * factor, bilinear, change_agg=True)
        self.outc = OutConv(32 * factor, n_classes)

        self.Change_head1 = Change_Head(256 * factor, type=op_type)
        self.Change_head2 = Change_Head(128 * factor, type=op_type)
        self.Change_head3 = Change_Head(64 * factor, type=op_type)

        if x_cross:
            # if use small backbone like ResNet18 or ResNet34, we recommend setting the num_head to 1
            self.CA1 = Cross_Attention(128 * factor, embed_dim=128 * factor, num_head=1, patch_size=8)
            self.CA2 = Cross_Attention(256 * factor, embed_dim=256 * factor, num_head=1, patch_size=4)
            self.CA3 = Cross_Attention(512 * factor, embed_dim=512 * factor, num_head=1, patch_size=2)
            self.CA4 = Cross_Attention(256 * factor, embed_dim=256 * factor, num_head=1, patch_size=4)
            self.CA5 = Cross_Attention(128 * factor, embed_dim=128 * factor, num_head=1, patch_size=8)

    def forward(self, befores, afters):
        if self.x_cross:
            befores[1], afters[1] = self.CA1(befores[1], afters[1])
            befores[2], afters[2] = self.CA2(befores[2], afters[2])
            befores[3], afters[3] = self.CA3(befores[3], afters[3])

        b_up4 = self.up1(befores[3], befores[2])
        b_up3 = self.up2(b_up4, befores[1])
        b_up2 = self.up3(b_up3, befores[0])

        a_up4 = self.up1(afters[3], afters[2])
        a_up3 = self.up2(a_up4, afters[1])
        a_up2 = self.up3(a_up3, afters[0])

        if self.x_cross:
            b_up4, a_up4 = self.CA4(b_up4, a_up4)
            b_up3, a_up3 = self.CA5(b_up3, a_up3)

        x4 = self.Change_head1(b_up4, a_up4)
        x3 = self.Change_head2(b_up3, a_up3)
        x2 = self.Change_head3(b_up2, a_up2)

        c_up2 = self.c_up2(x4, x3)
        c_up3 = self.c_up3(c_up2, x2)
        c = self.c_up4(c_up3)
        logits = self.outc(c)

        return logits
