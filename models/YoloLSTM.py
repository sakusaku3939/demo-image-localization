import torch
import torch.nn as nn


class YoloLSTM(nn.Module):
    def __init__(self):
        super(YoloLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=8192, hidden_size=3072, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(3072, 2)

    def forward(self, crops_in):
        crops_out = []

        for x in crops_in:
            crop_size, channels, height, width = x.size()

            # クロップ画像の処理
            x = x.view(crop_size, channels, height, width)
            x = self.cnn(x)
            x = x.view(crop_size, -1)

            # LSTM処理をして最後の出力を取得
            _, (h_n, _) = self.lstm(x)
            x = h_n[-1]

            # 2D座標を回帰
            x = self.fc(x)
            crops_out.append(x)

        crops_out = torch.stack(crops_out)
        return crops_out
