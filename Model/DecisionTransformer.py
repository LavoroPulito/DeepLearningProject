import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore


from .SelfAttention import TransformerBlock
from .Embedding import Embedding

class DecisionTransformer(nn.Module):
    def __init__(self,action_dim=192, d_model=256, n_heads=8, depth=6, max_turn=48):
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

        # predizioni separate per ogni feature dell'azione.
        # moltiplichiamo per 2 nella dimensione dell'output perché abbiamo 2 mosse da predire
        self.predict_player_user=nn.Linear(d_model, 2*2, bias=False)
        self.predict_slot_user=nn.Linear(d_model, 2*2, bias=False)
        self.predict_player_target=nn.Linear(d_model, 2*2, bias=False)
        self.predict_slot_target=nn.Linear(d_model, 2*2, bias=False)
        self.predict_mega=nn.Linear(d_model, 2*2, bias=False)
        self.predict_move=nn.Linear(d_model, 6*2, bias=False)
        
       # self.predict_action=nn.Linear(d_model, self.action_dim) #per predire le mosse (azioni discrete)

    
    def forward(self, state, move, battlefield, action, reward, turn, padding_mask=None):
        #x is a tensor of shape (batch_size, seq_length, d_model)
        batch_size=state['id'].size(0)

        x, stacked_padding_mask=self.token_embedding(state, move, battlefield, action, reward, turn, padding_mask) #embedding layer
        
        for block in self.tblocks:
            x=block(x, padding_mask=stacked_padding_mask) #transformer blocks

        
        x=x.reshape(batch_size, self.seq_length, 3, self.d_model).permute(0,2,1,3) #reshape to (batch_size, seq_length, 3, d_model)

        state_representation=x[:,1]

        def process_action(linear_layer, num_classes):
            out=linear_layer(state_representation)
            out=out.view(batch_size, self.seq_length, 2, num_classes)
            return F.log_softmax(out, dim=-1)

        
        
        p_user=process_action(self.predict_player_user, 2)
        s_user=process_action(self.predict_slot_user, 2)
        p_target=process_action(self.predict_player_target, 2)
        s_target=process_action(self.predict_slot_target, 2)
        mega=process_action(self.predict_mega, 2)
        move=process_action(self.predict_move, 6)

        preds=torch.cat([p_user, s_user, p_target, s_target, mega, move], dim=-1) 
        #avrà dimensioni (batch_size, self.seq_length, 2, 16=somma dim feature)
        
        
        return preds
            
            
        
        #action_preds=self.predict_action(state_representation) #linear layer to get logits for each action

        #return F.log_softmax(action_preds, dim=-1) #log softmax over the last dimension (num_tokens)
