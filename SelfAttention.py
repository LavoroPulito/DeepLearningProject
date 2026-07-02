import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        #Linear projection for query, key, value and output
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.unifyheads = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x, padding_mask=None): #x is our final_inputs
        batch_size, seq_length, d_model = x.size()

        #Linear projection and reshape
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)  
        k = self.key(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)    
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)  

        #Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  

        #Causal mask
        causal_mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)

        #Setting to -inf the positions that should be masked
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        #Padding mask with dimension (batch_size, seq_length)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat_interleave(self.n_heads, dim=0)  # Repeat the padding mask for each head
            padding_mask = padding_mask.unsqueeze(1)  
            #Setting to -inf the positions of fake turns (padding) that should be masked
            scores = scores.masked_fill(padding_mask == 0, float('-inf')) 

        #Softmax and weighted sum
        attention_weights = F.softmax(scores, dim=-1)
        #Wighted sum of values
        attention_output=torch.bmm(attention_weights, v).view(batch_size, self.n_heads, seq_length, self.head_dim).transpose(1,2).contiguous().view(batch_size, seq_length, d_model)
            

        return self.unifyheads(attention_output)


        

