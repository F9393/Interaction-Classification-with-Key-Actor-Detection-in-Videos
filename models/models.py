import torch.nn as nn
from .encoder import Encoder
from .decoder import FrameLSTM, EventLSTM

class Model1(nn.Module):
    def __init__(self, CNN_embed_dim = 256, h_frameLSTM = 256, h_eventLSTM = 256, num_classes = 8):
        super(Model1, self).__init__()

        self.encoder = Encoder(CNN_embed_dim = CNN_embed_dim)
        self.frameLSTM = FrameLSTM(CNN_embed_dim = CNN_embed_dim, h_frameLSTM = h_frameLSTM)
        self.eventLSTM = EventLSTM(h_frameLSTM = h_frameLSTM, h_eventLSTM = h_eventLSTM)

        self.fc = nn.Linear(in_features = h_eventLSTM, out_features = num_classes)

    def forward(self, x):

        out = self.encoder(x)

        f_out, (f_h_n, f_h_c) = self.frameLSTM(out)  

        e_out, (e_h_n, e_h_c) = self.eventLSTM(f_out)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out.reshape(-1,H)).view(B,T,-1) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

if __name__ == "__main__":
    import torch
    inp = torch.randn((2,5,3,224,224)) # (batch, time_step, channels, img_h, img_w)
    m1 = Model1(CNN_embed_dim = 10, h_frameLSTM = 4, h_eventLSTM = 6, num_classes = 8)
    out = m1(inp)
    print(out.shape)