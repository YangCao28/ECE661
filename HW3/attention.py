import torch.nn as nn
import torch
import math
import torch.nn.functional as F 
def get_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    mat = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mat= mat.masked_fill(mask[mask == 0], -1e9)
    attention = F.softmax(mat, dim = -1)
    if dropout is not None:
        attention = dropout(attention)
    return torch.matmul(attention, value), attention



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = None
    """
    Q: Implement the MultiHeadAttention's forward function. 
    Use the layers initialized in the __init__ function.
    Compute attention between query and key. 
    Apply the obtained attention scores on value.  
    Take care of the multi-head machenism. 
    Apply a fc layer to the output before return. 
    """
    def forward(self, query, key, value, mask=None):
        """
        Forward function
        :param query: [batch size, sequence length, hidden dim]
        :param key: [batch size, sequence length, hidden dim]
        :param value: [batch size, sequence length, hidden dim]
        :param mask: Just pass None to mask. No need to handle it specifically.
        :return: [batch size, sequence length, number of heads, hidden dim]
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batche_size = query.size(0)
        new_storage=[]
        for layer, item in zip(self.linear_layers, [query,key,value]):
            x = layer(item)
            x = x.view(batche_size,-1, self.h, self.d_k)

            x = x.permute(0,2,1,3)
            new_storage.append(x)
        query = new_storage[0]
        key = new_storage[1]
        value = new_storage[2]
        x,self.attention = get_attention(query,key,value,dropout=self.dropout) 

        x = x.view(batche_size, -1,self.h*self.d_k)
        output = self.output_linear(x)

        return output
