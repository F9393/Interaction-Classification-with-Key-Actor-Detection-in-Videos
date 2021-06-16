import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, CNN_embed_dim = 256, h_frameLSTM = 256, h_eventLSTM = 256, num_classes = 8):
        super(Decoder, self).__init__()

        self.CNN_embed_dim = CNN_embed_dim
        self.h_frameLSTM = h_frameLSTM                 # RNN hidden nodes
        self.h_eventLSTM = h_eventLSTM
        self.num_classes = num_classes

        self.frameLSTM = nn.LSTM(
            input_size = self.CNN_embed_dim,
            hidden_size = self.h_frameLSTM,        
            num_layers = 1,       
            batch_first = True,       # input & output has batch size as 1st dimension. i.e. (batch, time_step, input_size)
            bidirectional = True
        )

        self.eventLSTM = nn.LSTM(
            input_size = 2 * self.h_frameLSTM, # since input is from bidirectional LSTM
            hidden_size = self.h_eventLSTM,        
            num_layers = 1,       
            batch_first = True,       # input & output has batch size as 1st dimension. i.e. (batch, time_step, input_size)
            bidirectional = False
        )

        self.fc = nn.Linear(self.h_eventLSTM, self.num_classes)

    def forward(self, x):
        """ 
            f_out : shape (batch, time_step, 2 * hidden_size)
            f_h_n : shape (2 * n_layers, batch, hidden_size)
            f_h_c : shape (2 * n_layers, batch, hidden_size) 
            
            e_out : shape (batch, time_step, hidden_size)
            e_h_n : shape (n_layers, batch, hidden_size)
            e_h_c : shape (n_layers, batch, hidden_size) 
        """

        self.frameLSTM.flatten_parameters()
        f_out, (f_h_n, f_h_c) = self.frameLSTM(x, None)  

        self.eventLSTM.flatten_parameters()
        e_out, (e_h_n, e_h_c) = self.eventLSTM(f_out, None)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out.reshape(-1,H)).view(B,T,-1) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

if __name__ == "__main__":
    import torch
    inp = torch.randn((9,5,10)) # (batch,time_step,cnn_embedding_dim)
    dec = Decoder(CNN_embed_dim = 10, h_frameLSTM = 4, h_eventLSTM = 6)
    out = dec(inp)
    print(out.shape)