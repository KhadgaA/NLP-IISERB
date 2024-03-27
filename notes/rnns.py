import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
class StackRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,stacks = 2):
        super(StackRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 3*hidden_size
        self.stacks = stacks
        for layer in range(self.stacks):
            layer_input_size = input_size if layer == 0 else self.gate_size
            layer_out_size =  output_size if layer == (self.stacks-1) else self.gate_size
            w_i = nn.Linear(layer_input_size,hidden_size)
            u_i = nn.Linear(hidden_size,hidden_size)
            v_i = nn.Linear(hidden_size,layer_out_size)

            layer_params = (w_i,u_i,v_i)
            param_names = ['weight_ih_l{}', 'weight_hh_l{}','weight_hr_l{}']
            param_names = [x.format(layer) for x in param_names]
            for name,param in zip(param_names,layer_params):
                setattr(self,name,param)
            ic(param_names)

              
    def forward(self, x, hidden_state=None):
        for i in range(self.stacks):
            # ic(getattr(self,f'weight_ih_l{i}'))
            # ic(i)
            a = getattr(self,f'weight_ih_l{i}')(x)
            
            b = getattr(self,f'weight_hh_l{i}')(hidden_state)
            hidden_state = torch.tanh(a+b)
            output = getattr(self,f'weight_hr_l{i}')(hidden_state)
            x = output
        return output, hidden
    
    def init_hidden(self,batch):
        return nn.init.kaiming_uniform_(torch.empty(batch, self.hidden_size))


class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,stacks = 2):
        super(StackLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 3*hidden_size
        self.stacks = stacks
        for layer in range(self.stacks):
            layer_input_size = input_size if layer == 0 else self.gate_size
            layer_out_size =  output_size if layer == (self.stacks-1) else self.gate_size
            # forget gate
            wf_i = nn.Linear(layer_input_size,hidden_size)
            uf_i = nn.Linear(hidden_size,layer_input_size)
            # add gate
            wg_i = nn.Linear(layer_input_size,hidden_size)
            ug_i = nn.Linear(hidden_size,layer_input_size)

            wi_i = nn.Linear(layer_input_size,hidden_size)
            ui_i = nn.Linear(hidden_size,layer_input_size)

            # output gate
            wo_i = nn.Linear(layer_input_size,hidden_size)
            uo_i = nn.Linear(hidden_size,layer_input_size)

            # vf_i = nn.Linear(hidden_size,layer_out_size)

            layer_params = (wf_i,uf_i,wg_i,ug_i,wi_i,ui_i,wo_i,uo_i)
            param_names = ['weight_ih_f_l{}', 'weight_hh_f_l{}','weight_ih_g_l{}', 'weight_hh_g_l{}','weight_ih_i_l{}', 'weight_hh_i_l{}','weight_ih_o_l{}', 'weight_hh_o_l{}']
            param_names = [x.format(layer) for x in param_names]
            for name,param in zip(param_names,layer_params):
                setattr(self,name,param)
            ic(param_names)

              
    def forward(self, x, hidden_state=None,context=None):
        

        for i in range(self.stacks):
            hidden_state = getattr(self,f'hidden_state_l{i}') ##### here the shape and init of hidden state has to be explicitly declared in init func.
            context = getattr(self,f'context_l{i}')
            # forget gate
            ft = torch.sigmoid(getattr(self,f'weight_ih_f_l{i}')(x) + getattr(self,f'weight_hh_f_l{i}')(hidden_state))
            kt = torch.mul(context,ft)
            # add gate
            gt = torch.tanh(getattr(self,f'weight_ih_g_l{i}')(x) + getattr(self,f'weight_hh_g_l{i}')(hidden_state))
            it = torch.sigmoid(getattr(self,f'weight_ih_i_l{i}')(x) + getattr(self,f'weight_hh_i_l{i}')(hidden_state))
            jt = torch.mul(gt,it)
            context = jt + kt 
            # output gate
            ot = torch.sigmoid(getattr(self,f'weight_ih_o_l{i}')(x) + getattr(self,f'weight_hh_o_l{i}')(hidden_state))
            hidden_state = torch.mul(ot,torch.tanh(context))
            x = hidden_state
        return hidden_state , context
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))




if __name__=='__main__':
    import numpy as np
    data = torch.randint(5,size = (5,5))
    hidden = torch.randint(5,size = (5,10))
    # ic(data)
    model = StackLSTM(5,10,5,stacks=2).float()
    out,hd = model(data.float(),hidden.float())
    ic(out,hd)