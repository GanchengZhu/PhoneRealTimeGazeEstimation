import torch
import torch.nn as nn


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel_num, compress_rate):
        super(SqueezeExcitationLayer, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_excitation = nn.Sequential(nn.Linear(channel_num, channel_num // compress_rate, bias=True),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(channel_num // compress_rate, channel_num, bias=True),
                                                nn.Sigmoid())

    def forward(self, feature_input):
        batch_size, num_channels, _, _ = feature_input.size()
        pooling_out = self.pooling(feature_input)
        pooling_out = pooling_out.view(pooling_out.size(0), -1)
        out = self.squeeze_excitation(pooling_out)
        output_tensor = torch.mul(feature_input, out.view(batch_size, num_channels, 1, 1))
        return output_tensor


class LABNLayer(nn.Module):
    def __init__(self, input_size, channels):
        super(LABNLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, channels * 2), nn.LeakyReLU())
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x, factor):
        style = self.fc(factor)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = self.batch_norm(x)
        x = x * (style[:, 0, :, :, :] + 1.) + style[:, 1, :, :, :]
        return x


class EyeBranchModel(nn.Module):
    def __init__(self):
        super(EyeBranchModel, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.se_block_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SqueezeExcitationLayer(48, 16),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.down_sampling = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.se_block_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SqueezeExcitationLayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

        self.labn_48 = LABNLayer(128, 48)
        self.labn_64_a = LABNLayer(128, 64)
        self.labn_128 = LABNLayer(128, 128)
        self.labn_64_b = LABNLayer(128, 64)

    def forward(self, x, factor):
        x = self.conv_block(x)
        x = self.labn_48(x, factor)
        x = self.se_block_1(x)
        x = self.labn_64_a(x, factor)
        x_1 = self.down_sampling(x)

        x_2 = self.conv(x_1)
        x_2 = self.labn_128(x_2, factor)
        x_2 = self.se_block_2(x_2)
        x_2 = self.labn_64_b(x_2, factor)
        x_2 = self.relu(x_2)

        return torch.cat((x_1, x_2), 1)


class FaceBranchModel(nn.Module):
    def __init__(self):
        super(FaceBranchModel, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            SqueezeExcitationLayer(192, 16),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitationLayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SqueezeExcitationLayer(64, 16),
        )
        self.two_fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main_branch(x)
        x = x.view(x.size(0), -1)
        x = self.two_fc(x)
        return x


class MGazeNet(nn.Module):
    def __init__(self):
        super(MGazeNet, self).__init__()
        self.eye_branch = EyeBranchModel()
        self.eye_se_block_a = nn.Sequential(
            SqueezeExcitationLayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.labn_layer = LABNLayer(128, 64)
        self.eye_se_block_b = nn.Sequential(
            nn.ReLU(inplace=True),
            SqueezeExcitationLayer(64, 16)
        )
        self.face_branch = FaceBranchModel()

        self.eye_fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )
        # Gaze Regression
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.rect_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, left_eye, right_eye, face, rect):
        out_face = self.face_branch(face)
        out_rect = self.rect_fc(rect)
        factor = torch.cat((out_face, out_rect), 1)
        out_left_eye = self.eye_branch(left_eye, factor)
        out_right_eye = self.eye_branch(right_eye, factor)
        out_eyes = torch.cat((out_left_eye, out_right_eye), 1)
        out_eyes = self.eye_se_block_a(out_eyes)
        out_eyes = self.labn_layer(out_eyes, factor)
        out_eyes = self.eye_se_block_b(out_eyes)
        out_eyes = out_eyes.view(out_eyes.size(0), -1)
        out_eyes = self.eye_fc(out_eyes)
        gaze_feature = torch.cat((out_eyes, out_face, out_rect), 1)
        gaze = self.fc(gaze_feature)

        return torch.cat((gaze, gaze_feature), 1)


if __name__ == '__main__':
    m = MGazeNet()

    feature_dict = {"faceImg": torch.zeros(6, 3, 224, 224),
                    "leftEyeImg": torch.zeros(6, 3, 112, 112),
                    "rightEyeImg": torch.zeros(6, 3, 112, 112),
                    "faceGridImg": torch.zeros(6, 12),
                    "label": torch.zeros(6, 2), "frame": "test.jpg"}
    a = m(feature_dict["leftEyeImg"], feature_dict["rightEyeImg"], feature_dict["faceImg"], feature_dict["faceGridImg"])
    print(a.shape)
