import torch
import torch.nn as nn
from collections import OrderedDict

class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
        self.load_weight()
        self.freeze()

    def load_weight(self, checkpoint_path='checkpoints/3000000-BL.ckpt'):
        c_checkpoint = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        self.load_state_dict(new_state_dict)

    def freeze(self):
        for param in self.lstm.parameters():
            param.requires_grad = False

        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized


    
