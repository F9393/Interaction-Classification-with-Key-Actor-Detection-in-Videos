import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 2D CNN encoder
class Encoder(nn.Module):
    def __init__(self, CNN_embed_dim = 256):
        """
            Load the pretrained ResNet-152 and replace last fc layer with embedding layer.
            Only embedding layer is trainable.

        """
        super(Encoder, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.resnet.eval() ## use pretrained calculated running_stats

        self.embedding_layer = nn.Linear(resnet.fc.in_features, CNN_embed_dim)

        # initialize embedding layer
        # nn.init.constant_(self.embedding_layer.bias, 0.0)
        # nn.init.xavier_normal_(self.embedding_layer.weight)
        
    def forward(self, x):
        # cnn_embed_seq = []
        # for t in range(x.size(1)):        
        #     with torch.no_grad():
        #         out = self.resnet(x[:, t, :, :, :])     # ResNet
        #         out = out.view(out.size(0), -1)         # flatten output of conv

        #     out = self.embedding_layer(out)                
        #     out= F.relu(out)

        #     cnn_embed_seq.append(out)

        # # swap time and batch dimensions. Resulting shape is (batch,frames,embed_dim)
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1) 

        # return cnn_embed_seq

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
