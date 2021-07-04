import torch.nn as nn

class FrameLSTM(nn.Module):
    def __init__(self, input_size = 256, h_frameLSTM = 256):
        super(FrameLSTM, self).__init__()

        self.input_size = input_size
        self.h_frameLSTM = h_frameLSTM                

        self.frameLSTM = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.h_frameLSTM,        
            num_layers = 1,       
            batch_first = True,       # input & output has batch size as 1st dimension. i.e. (batch, time_step, input_size)
            bidirectional = True
        )

        # for name, param in self.frameLSTM.named_parameters(): # use xavier_normal initilization
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.orthogonal_(param)

        for layer in self.frameLSTM._all_weights:
            for name in layer: 
                if 'weight' in name:
                    weight = getattr(self.frameLSTM, name)
                    nn.init.xavier_normal_(weight.data)
                elif 'bias' in name:
                    bias = getattr(self.frameLSTM, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)

    def forward(self, x):
        """ 
            f_out : shape (batch, time_step, 2 * hidden_size)
            f_h_n : shape (2 * n_layers, batch, hidden_size)
            f_h_c : shape (2 * n_layers, batch, hidden_size) 
        """

        self.frameLSTM.flatten_parameters()
        f_out, (f_h_n, f_h_c) = self.frameLSTM(x, None)  

        return f_out, (f_h_n, f_h_c)


class EventLSTM(nn.Module):
    def __init__(self, input_size = 512, h_eventLSTM = 256):
        super(EventLSTM, self).__init__()

        self.input_size = input_size         
        self.h_eventLSTM = h_eventLSTM       

        self.eventLSTM = nn.LSTM(
            input_size = self.input_size, # since input is from bidirectional LSTM
            hidden_size = self.h_eventLSTM,        
            num_layers = 1,       
            batch_first = True,       # input & output has batch size as 1st dimension. i.e. (batch, time_step, input_size)
            bidirectional = False
        )

        # for name, param in self.eventLSTM.named_parameters(): # use xavier_normal initilization
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.orthogonal_(param)

        for layer in self.eventLSTM._all_weights:
            for name in layer: 
                if 'weight' in name:
                    weight = getattr(self.eventLSTM, name)
                    nn.init.xavier_normal_(weight.data)
                elif 'bias' in name:
                    bias = getattr(self.eventLSTM, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)

    def forward(self, x):
        """ 
            e_out : shape (batch, time_step, hidden_size)
            e_h_n : shape (n_layers, batch, hidden_size)
            e_h_c : shape (n_layers, batch, hidden_size) 
        """

        self.eventLSTM.flatten_parameters()
        e_out, (e_h_n, e_h_c) = self.eventLSTM(x, None)

        return e_out, (e_h_n, e_h_c)