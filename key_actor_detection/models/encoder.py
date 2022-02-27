import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 2D CNN encoder
class Encoder(nn.Module):
    def __init__(self, CNN_embed_dim = 256):
        """
            Load pretrained ResNet-152 and replace last fc layer with embedding layer.
            Only embedding layer is trainable.

        """
        super(Encoder, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        # resnet = models.vgg16(pretrained=True)
        # modules = list(resnet.children())[:-1]
             # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.resnet.eval() ## use pretrained calculated running_stats
        self.embedding_layer = nn.Linear(resnet.fc.in_features, CNN_embed_dim)
        # self.embedding_layer = nn.Linear(25088, CNN_embed_dim)

        # initialize embedding layer
        # nn.init.constant_(self.embedding_layer.bias, 0.0)
        # nn.init.xavier_normal_(self.embedding_layer.weight)
        
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.view(-1,C,H,W)      
        with torch.no_grad():  
            out = self.resnet(x)    
        out = torch.squeeze(out)
        out = self.embedding_layer(out)  
        out = F.relu(out)
        out = out.view(B,T,-1) 
        return out


if __name__ == "__main__":
    t = torch.randn((2,5,3,224,224)) # (batch, time_step, channels, img_h, img_w)
    encoder = Encoder()
    out = encoder(t)

    print(out.shape)
