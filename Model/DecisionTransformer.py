import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
try:
    from .SelfAttention import TransformerBlock
    from .Embedding import Embedding
except ImportError:
    from SelfAttention import TransformerBlock
    from Embedding import Embedding


class DecisionTransformer(nn.Module):
    def __init__(self, action_dim=360, d_model=256, n_heads=8, depth=6, max_turn=49, dropout=0.1):
        super().__init__()
        self.action_dim = action_dim #all the possible actions
        self.d_model=d_model
        self.seq_length=max_turn
        max_seq_length=3*max_turn #max_seq_length is 3 times the number of tokens (R,s,a) in each game 

        #Embedding layer for states, actions, and rewards
        self.token_embedding = Embedding(d_model=d_model, max_turn=max_turn)

        # Transformer blocks
        self.tblocks=nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_seq_length=max_seq_length
                ) for _ in range(depth)
        ])
        
        self.predict_action=nn.Linear(d_model, 2*self.action_dim) #per predire il token dell'azione
    
    def forward(self, state, move, battlefield, action, reward, turn, padding_mask=None, legal_action_mask=None):
        
        batch_size=state['id'].size(0) #number of real turn in game

        #x is a tensor of shape (batch_size, seq_length, d_model)
        x, stacked_padding_mask=self.token_embedding(state, move, battlefield, action, reward, turn, padding_mask) #embedding layer
        
        for block in self.tblocks:
            x=block(x, padding_mask=stacked_padding_mask) #transformer blocks
        
        x=x.reshape(batch_size, self.seq_length, 3, self.d_model).permute(0,2,1,3) #reshape to (batch_size, 3, seq_length, d_model)

        state_representation=x[:,1] #[state,action,reward] -> [state]. Just the state column (batch_size, seq_length, d_model) 
        
        action_preds=self.predict_action(state_representation) #linear layer to get logits for each action

        action_preds=action_preds.view(batch_size, self.seq_length, 2, self.action_dim) 

        if legal_action_mask is not None:
            # legal_action_mask: bool, shape (batch_size, seq_length, action_dim)
            # True = legal action. The actions share the same legal action mask. They are related to the same state.
            mask = legal_action_mask.unsqueeze(2)  # (batch, seq, 1, action_dim) -> broadcast on dim=2 (the two heads)
            action_preds = action_preds.masked_fill(~mask, float('-inf'))
            
        return F.log_softmax(action_preds, dim=-1) #log softmax over the last dimension (num_tokens)


class AmpDecisionTransformer(DecisionTransformer):
    """Autocast INSIDE the forward: With nn.DataParallel,
    each replica runs in a separate thread 
    and the external autocast context does not propagate."""
    def forward(self, *args, **kwargs):
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            return super().forward(*args, **kwargs)