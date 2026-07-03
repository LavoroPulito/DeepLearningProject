import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, max_seq_length=128):
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

        #casual mask
        causal_mask=torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer("causal_mask", causal_mask)


    def forward(self, x, padding_mask=None): #x is our final_inputs
        batch_size, seq_length, d_model = x.size() 
        #batch_size=numero partite, seq_length=3*numero turni (R,s,a)

        #Linear projection and reshape
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)  
        k = self.key(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)    
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1,2).contiguous().view(batch_size * self.n_heads, seq_length, self.head_dim)  

        #Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  

        #Causal mask (tagliamo la causal mask alla lunghezza attuale della sequenza)
        causal_mask = self.causal_mask[:seq_length, :seq_length]

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
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.0, max_seq_length=128):
        super().__init__()
        self.self_attention = SelfAttention(d_model, n_heads, max_seq_length)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), 
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Self-attention
        attention_output = self.self_attention(self.layer_norm1(x), padding_mask)
        x = x + self.dropout(attention_output) #residual connection
        

        # Feed-forward
        feed_forward_output = self.feed_forward(self.layer_norm2(x))
        x = x + feed_forward_output #residual connection

        return x



        

