from torch import nn

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # b, 64, 64, 64
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 64, 32, 32
            nn.Conv2d(64, 32, 3, padding=1),  # b, 32, 32, 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 32, 16, 16
            nn.Conv2d(32, 16, 3, padding=1),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 8, 8
            nn.Flatten()
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 8, 8)), #b, 16, 8, 8
            nn.ConvTranspose2d(16, 32, 3, stride=2),  # b, 32, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1),  # b, 64, x, x
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1),  # b, 3, x, x
            CropLayer(),
            nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs

class PrintLayer(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def forward(self, inputs):
        print(self.prefix, inputs.shape)
        return inputs

class CropLayer(nn.Module):
    def forward(self, inputs):
        return inputs[:, :, :64, :64]

class ConvED(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, inputs):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs