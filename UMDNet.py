import torch
from torch import nn
import torch.nn.functional as F
import mix_transformer
from modules import DSM
from modules import MCAM


def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class GAM(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(GAM, self).__init__()
        self.ch2 = ch_2
        self.conv_pre = convblock(ch_1, ch_2, 3, 1, 1)

    def forward(self, rgb, pre):
        cur_size = rgb.size()[2:]

        pre = self.conv_pre(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))

        fus = pre + rgb

        return fus


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.d3 = GAM(512, 320)
        self.d2 = GAM(320, 128)
        self.d1 = GAM(128, 64)

        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_3 = nn.Conv2d(320, 1, 1, 1, 0)
        self.score_4 = nn.Conv2d(512, 1, 1, 1, 0)

    def forward(self, rgb):

        d4 = rgb[3]
        d3 = self.d3(rgb[2], d4)
        d2 = self.d2(rgb[1], d3)
        d1 = self.d1(rgb[0], d2)

        score1 = self.score_1(d1)
        score2 = self.score_2(d2)
        score3 = self.score_3(d3)
        score4 = self.score_4(d4)

        return score1, score2, score3, score4


class CI(nn.Module):
    def __init__(self, ch_2, n_split):  # ch_1:previous, ch_2:current/output
        super(CI, self).__init__()

        self.n_split = n_split
        self.ch_2 = ch_2
        split_dim = self.ch_2 // self.n_split
        self.sal_blocks = nn.ModuleList([nn.Conv2d(split_dim, 1, 1, 1, 0) for i in range(n_split)])

    def forward(self, x):

        # scores = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).cuda()

        split_dim = self.ch_2 // self.n_split
        split_features = []
        scores = []
        for i in range(self.n_split):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            # tsets = x[:, start_idx:end_idx, :, :]
            # print(tsets.shape)
            split_features.append(x[:, start_idx:end_idx, :, :])
            # split_features[i] = x[:, start_idx:end_idx, :, :]
            # scores += self.sal_blocks[i](split_features[i])
            scores.append(self.sal_blocks[i](split_features[i]))
        # print(scores[0].shape)
        scores = sum(scores)
        # print(scores.shape)

        return scores


class Decoder_d(nn.Module):
    def __init__(self):
        super(Decoder_d, self).__init__()

        self.d3 = GAM(512, 320)
        self.d2 = GAM(320, 128)
        self.d1 = GAM(128, 64)

        self.c4 = CI(512, 4)
        self.c3 = CI(320, 4)
        self.c2 = CI(128, 4)
        self.c1 = CI(64, 4)


    def forward(self, rgb):

        d4 = rgb[3]
        d3 = self.d3(rgb[2], d4)
        d2 = self.d2(rgb[1], d3)
        d1 = self.d1(rgb[0], d2)

        score1 = self.c1(d1)
        score2 = self.c2(d2)
        score3 = self.c3(d3)
        score4 = self.c4(d4)

        return score1, score2, score3, score4





class Segformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        # self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]
        self.encoder = getattr(mix_transformer, backbone)()
        ## initilize encoder
        if pretrained:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

    def forward(self):
        model = Segformer('mit_b3', pretrained=True)
        return model


class Mnet(nn.Module):
    def __init__(self, backbone="mit_b3", pretrained=True):
        super(Mnet, self).__init__()

        net = Segformer(backbone, pretrained)
        self.rgb_encoder = net.encoder
        self.decoder_n = Decoder()
        self.decoder_d = Decoder_d()
        self.sigmoid = nn.Sigmoid()
        self.DSM1 = DSM(64)
        self.DSM2 = DSM(128)
        self.DSM3 = DSM(320)
        self.MCAM = MCAM(512,512)

    def forward(self, rgb, data_label):
        # rgb
        B = rgb.shape[0]
        H = rgb.shape[2]
        rgb_f = []

        # stage 1
        x, H, W = self.rgb_encoder.patch_embed1(rgb)
        for i, blk in enumerate(self.rgb_encoder.block1):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.DSM1(x)
        rgb_f.append(x)

        # stage 2
        x, H, W = self.rgb_encoder.patch_embed2(x)
        for i, blk in enumerate(self.rgb_encoder.block2):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.DSM2(x)
        rgb_f.append(x)

        # stage 3
        x, H, W = self.rgb_encoder.patch_embed3(x)
        for i, blk in enumerate(self.rgb_encoder.block3):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.DSM3(x)
        rgb_f.append(x)

        # stage 4
        x, H, W = self.rgb_encoder.patch_embed4(x)
        for i, blk in enumerate(self.rgb_encoder.block4):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm4(x)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.MCAM(x)
        rgb_f.append(x)

        if data_label == 1:
            score1, score2, score3, score4 = self.decoder_n(rgb_f)
        else:
            score1, score2, score3, score4 = self.decoder_d(rgb_f)


        score1 = F.interpolate(score1, rgb.shape[2:], mode='bilinear', align_corners=True)
        score2 = F.interpolate(score2, rgb.shape[2:], mode='bilinear', align_corners=True)
        score3 = F.interpolate(score3, rgb.shape[2:], mode='bilinear', align_corners=True)
        score4 = F.interpolate(score4, rgb.shape[2:], mode='bilinear', align_corners=True)


        return score1, score2, score3, score4, self.sigmoid(score1), self.sigmoid(score2), \
               self.sigmoid(score3), self.sigmoid(score4)
