import torch
import torch.nn as nn

#this define a basic block for CNN , padding is set to reflect because it maintains spatial info

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):#3->64 ,64->128,128->256,256->52
        super().__init__()
        self.initial = nn.Sequential(#the paper follows first making a initial conv layer
            nn.Conv2d(
                in_channels * 2,#2 times becasue we are sending x(sat image),y(maps image) concatenated along the channels
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                #for the last layer stride is set to 1 , according to paper
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
            
        #to generate a single feature map(single channel) representing the real fake classification
        #without this layer it wouldnt be able to identify the generated feature maps due to multiple feature maps
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)#concatenated along the channels
        x = self.initial(x)
        x = self.model(x)
        return x

#a test case to check code is working or not
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
