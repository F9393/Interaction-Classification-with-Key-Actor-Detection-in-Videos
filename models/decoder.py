import torch.nn as nn

class FrameLSTM(nn.Module):
    """
        Frame-Level LSTM

        Parameters
        ----------
        module_input: tensor
            shape : (B,T,I) #(batch,time_step,input_dim)

        Returns
        -------
        module_output : tuple 
            f_out, (f_h_n. f_h_c)

            f_out : shape (batch, time_step, 2 * hidden_size)
            f_h_n : shape (2 * n_layers, batch, hidden_size)
            f_h_c : shape (2 * n_layers, batch, hidden_size) 

    """
    def __init__(self, input_size, hidden_size, winit = None, forget_gate_bias = None, **kwargs):
        super(FrameLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size                

        self.frameLSTM = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,        
            num_layers = 1,       
            batch_first = True,     
            bidirectional = True
        )

        for layer in self.frameLSTM._all_weights:
            for name in layer: 
                if 'weight' in name:
                    weight = getattr(self.frameLSTM, name)
                    if winit=="xavier_normal":
                        nn.init.xavier_normal_(weight.data)
                elif 'bias' in name:
                    bias = getattr(self.frameLSTM, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    if forget_gate_bias is not None and forget_gate_bias != 'default':
                        bias.data[start:end].fill_(forget_gate_bias*1.0/2)
                            

    def forward(self, x, init_hidden = None):
        """ 
            f_out : shape (batch, time_step, 2 * hidden_size)
            f_h_n : shape (2 * n_layers, batch, hidden_size)
            f_h_c : shape (2 * n_layers, batch, hidden_size) 
        """

        self.frameLSTM.flatten_parameters()
        f_out, (f_h_n, f_h_c) = self.frameLSTM(x, init_hidden)  

        return f_out, (f_h_n, f_h_c)


class EventLSTM(nn.Module):
    """
        Event-Level LSTM

        Parameters
        ----------
        module_input: tensor
            shape : (B,T,I) #(batch,time_step,input_dim)

        Returns
        -------
        module_output : tuple 
            e_out, (e_h_n. e_h_c)

            e_out : shape (batch, time_step, hidden_size)
            e_h_n : shape (n_layers, batch, hidden_size)
            e_h_c : shape (n_layers, batch, hidden_size) 

    """
    def __init__(self, input_size, hidden_size, winit = None, forget_gate_bias = None, **kwargs):
        super(EventLSTM, self).__init__()

        self.input_size = input_size         
        self.hidden_size = hidden_size       

        self.eventLSTM = nn.LSTM(
            input_size = self.input_size, 
            hidden_size = self.hidden_size,        
            num_layers = 1,       
            batch_first = True,     
            bidirectional = False
        )

        for layer in self.eventLSTM._all_weights:
            for name in layer: 
                if 'weight' in name:
                    weight = getattr(self.eventLSTM, name)
                    if winit=="xavier_normal":
                        nn.init.xavier_normal_(weight.data)
                elif 'bias' in name:
                    bias = getattr(self.eventLSTM, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    if forget_gate_bias is not None and forget_gate_bias != 'default':
                        bias.data[start:end].fill_(forget_gate_bias*1.0/2)
                            

    def forward(self, x, init_hidden = None):
        """ 
            e_out : shape (batch, time_step, hidden_size)
            e_h_n : shape (n_layers, batch, hidden_size)
            e_h_c : shape (n_layers, batch, hidden_size) 
        """

        self.eventLSTM.flatten_parameters()
        e_out, (e_h_n, e_h_c) = self.eventLSTM(x, init_hidden)

        return e_out, (e_h_n, e_h_c)