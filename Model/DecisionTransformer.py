#Decision Transformer

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


from .SelfAttention import TransformerBlock
from .Embedding import Embedding

class DecisionTransformer(nn.Module):
    def __init__(self,action_dim=384, d_model=256, n_heads=8, depth=6, max_turn=49):
        super().__init__()
        self.action_dim = action_dim #tutte le possibili azioni (mosse)
        self.d_model=d_model
        self.seq_length=max_turn
        max_seq_length=3*max_turn #max_seq_length is 3 times the number of tokens (R,s,a) for each turn

        #Embedding layer for states, actions, and rewards
        self.token_embedding = Embedding(d_model=d_model)

        # Transformer blocks
        self.tblocks=nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=0.0,
                max_seq_length=max_seq_length
                ) for _ in range(depth)
        ])
        
        self.predict_action=nn.Linear(d_model, 2*self.action_dim) #per predire il token dell'azione
    
    def forward(self, state, move, battlefield, action, reward, turn, padding_mask=None):
        #x is a tensor of shape (batch_size, seq_length, d_model)
        batch_size=state['id'].size(0)

        x, stacked_padding_mask=self.token_embedding(state, move, battlefield, action, reward, turn, padding_mask) #embedding layer
        
        for block in self.tblocks:
            x=block(x, padding_mask=stacked_padding_mask) #transformer blocks

        
        x=x.reshape(batch_size, self.seq_length, 3, self.d_model).permute(0,2,1,3) #reshape to (batch_size, 3, seq_length, d_model)

        state_representation=x[:,1] #prendo solo la componente di stato di x. ha dimensione (batch_size, seq_length, d_modele) 
        
        action_preds=self.predict_action(state_representation) #linear layer to get logits for each action
        #va fatta una maschera per le azioni illegali da applicare ad action_preds la maschera mette le azioni illegali di action_preds a -inf
        action_preds=action_preds.view(batch_size, self.seq_length, 2, self.action_dim) #dipende da come scegliamo di fare la loss
        return F.log_softmax(action_preds, dim=-1) #log softmax over the last dimension (num_tokens)
