import torch
from torch import nn

class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvReluPool(3, 64), # C1 => bx64x32x32
            ConvReluPool(64, 32), # C2 => bx32x16x16
            ConvReluPool(32, 16), # C3 => bx16x8x8
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
            # DeconvRelu(16, 32), # D1 => 32x*x*
            # DeconvRelu(32, 64), # D2 => 64x*x*
            # DeconvRelu(64, 3), # D3 => 3x*x*
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

class ConvReluPool(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
    
    def forward(self, inputs):
        return self.layer(inputs)

class ConvNormRelu(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_features, out_features,
            kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(True)
            # nn.ReLU(True)
        )
    
    def forward(self, inputs):
        return self.layer(inputs)

class DeconvNormRelu(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features,
            kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, inputs):
        return self.layer(inputs)

class DeconvRelu(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2),  # b, 32, 8, 8
            nn.ReLU(True),
        )
    
    def forward(self, inputs):
        return self.layer(inputs)

class ConvED(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, inputs):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs

class LeakyConvED(nn.Module):
    def __init__(self):
        super().__init__()
        a = 128
        b = 64
        c = 16
        # d = 16

        out_size = a
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(out_size, out_size, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
            nn.Conv2d(out_size, out_channels= 3,
                      kernel_size= 3, padding= 1),
            nn.Tanh()
        )

        self.encoder = nn.Sequential(
            ConvNormRelu(3, a),
            ConvNormRelu(a, b),
            ConvNormRelu(b, c),
            nn.Flatten(),
        #     ConvNormRelu(c, d),
        )

        self.decoder = nn.Sequential(
        #     DeconvNormRelu(d, c),
            nn.Unflatten(1, (16,8,8)),
            DeconvNormRelu(c, b),
            DeconvNormRelu(b, a),
            self.final_layer
        )
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        outputs = self.decoder(x)
        return outputs

class SkippyConvED(nn.Module):
    def __init__(self):
        super().__init__()
        a = 128
        b = 64
        c = 16
        d = 8

        out_size = 2*a
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(out_size, out_size, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(),
            nn.Conv2d(out_size, out_channels= 3,
                      kernel_size= 3, padding= 1),
            nn.Tanh()
        )

        self.fl = nn.Flatten()
        self.c1 = ConvNormRelu(3, a)
        self.sk1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 786),
        )
        self.c2 = ConvNormRelu(a, b)
        self.sk2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 98),
        )
        self.c3 = ConvNormRelu(b, c)
        self.sk3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 12),
        )
        self.c4 = nn.Sequential(
            ConvNormRelu(c, d),
            nn.Flatten()
        )

        self.d1 = nn.Sequential(
            nn.Unflatten(1, (8,4,4)),
            DeconvNormRelu(d, c)
            )
        self.us1 = nn.Sequential(
            nn.Linear(12, 1024),
            nn.Unflatten(1, (16,8,8)),
        )
        self.d2 = DeconvNormRelu(2*c, b)
        self.us2 = nn.Sequential(
            nn.Linear(98, 16384),
            nn.Unflatten(1, (64,16,16)),
        )
        self.d3 = DeconvNormRelu(2*b, a)
        self.us3 = nn.Sequential(
            nn.Linear(786, 131072),
            nn.Unflatten(1, (128,32,32)),
        )
        self.d4 = self.final_layer
    
    def encode(self, inputs):
        x = self.c1(inputs)
        c1_out = self.sk1(x)
        x = self.c2(x)
        c2_out = self.sk2(x)
        x = self.c3(x)
        c3_out = self.sk3(x)
        x = self.c4(x)
        outputs = torch.cat((x, c1_out, c2_out, c3_out), dim=1)
        return outputs
    
    def decode(self, inputs):
        x1 = torch.narrow(inputs, 1, 0, 128)
        c1_out = torch.narrow(inputs, 1, 128, 786)
        c2_out = torch.narrow(inputs, 1, 914, 98)
        c3_out = torch.narrow(inputs, 1, 1012, 12)
        x = self.d1(x1)
        z = torch.cat((x, self.us1(c3_out)), dim=1)
        x = self.d2(torch.cat((x, self.us1(c3_out)), dim=1))
        x = self.d3(torch.cat((x, self.us2(c2_out)), dim=1))
        return self.d4(torch.cat((x, self.us3(c1_out)), dim=1))

    
    def forward(self, inputs):
        x = self.encode(inputs)
        outputs = self.decode(x)
        return outputs