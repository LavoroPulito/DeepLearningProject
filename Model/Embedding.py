# ── Embedding ────────────────────────────────────────
"""The IDs (Pokemon, move, ability, item) arrive ALREADY remapped 
to the embedding indices from the Dataset (see id_maps.py / preprocess.py): 
index 0 = unknown/padding. No dictionary lookup is performed here.
"""
import torch  # type: ignore
import torch.nn as nn  # type: ignore

try:
    from .id_maps import N_POKE, N_MOVE, N_ABILITY, N_ITEM
except ImportError:
    from id_maps import N_POKE, N_MOVE, N_ABILITY, N_ITEM


class Embedding(nn.Module):
    def __init__(self, d_model=256, feat_dim=16, max_turn=49):
        super().__init__()
        self.d_model = d_model
        # --- state: 12 pokemon + field  

        # POKEMON. features: (player, id, type1, type2, ability, item, slot, stats, stats_change, status, [moves]x4, hp_ratio)
        # --- discrete features  
        self.embed_player = nn.Embedding(2, feat_dim)
        self.embed_id = nn.Embedding(N_POKE, feat_dim)        # 242
        self.embed_type1 = nn.Embedding(19, feat_dim)
        self.embed_type2 = nn.Embedding(19, feat_dim)
        self.embed_ability = nn.Embedding(N_ABILITY, feat_dim)  # 150
        self.embed_item = nn.Embedding(N_ITEM, feat_dim)        # 40
        self.embed_slot = nn.Embedding(5, feat_dim)

        # --- continuous features 
        self.embed_stats = nn.Linear(6, feat_dim)
        self.embed_stats_change = nn.Linear(5, feat_dim)
        self.embed_status = nn.Linear(6, feat_dim)
        self.embed_hp_ratio = nn.Linear(1, feat_dim)

        # MOVE. features: (id, type, damage class, target_class, power, priority, accuracy)
        # --- discrete features
        self.embed_id_move = nn.Embedding(N_MOVE, feat_dim)   # 323
        self.embed_d_class = nn.Embedding(4, feat_dim)        # 0..3
        self.embed_move_type = nn.Embedding(19, feat_dim)
        self.embed_t_class = nn.Embedding(17, feat_dim)       # 0..16

        # --- continuous features  
        self.embed_power = nn.Linear(1, feat_dim)
        self.embed_priority = nn.Linear(1, feat_dim)
        self.embed_accuracy = nn.Linear(1, feat_dim)

        # FIELD. features: (current weather, speed modifier)
        self.embed_current_weather = nn.Embedding(5, feat_dim)
        self.embed_speed_modifier = nn.Linear(3, feat_dim)

        # complete state dimension: 
        # 12 pokemon x 11 feature + 12 x 4 moves x 7 feature + 2 field features
        state_in = (12 * 11 + 12 * 4 * 7 + 2) * feat_dim   # 7520 using feat_dim=16
        self.state_proj = nn.Linear(state_in, d_model)

        # --- action (2 action each turn x 6 components)
        self.embed_player_user = nn.Embedding(2, feat_dim)
        self.embed_slot_user = nn.Embedding(3, feat_dim)      # 0=pass, 1, 2
        self.embed_player_target = nn.Embedding(2, feat_dim)
        self.embed_slot_target = nn.Embedding(5, feat_dim)
        self.embed_mega = nn.Embedding(2, feat_dim)
        self.embed_move = nn.Embedding(6, feat_dim)
        self.action_proj = nn.Linear(2 * 6 * feat_dim, d_model)

        # --- reward (return-to-go 0/1) and turn (simil timestamp)
        self.embed_reward = nn.Embedding(2, d_model)
        self.embed_turn = nn.Embedding(max_turn, d_model)

    def forward(self, state, move, battlefield, action, reward, turn,
                padding_mask=None):
        batch_size, seq_len = state['id'].shape[:2]

        # --- state
        pokemon_emb = torch.cat([
            self.embed_player(state['player']),
            self.embed_id(state['id']),
            self.embed_type1(state['type1']),
            self.embed_type2(state['type2']),
            self.embed_ability(state['ability']),
            self.embed_item(state['item']),
            self.embed_slot(state['slot']),
            self.embed_stats(state['stats']),
            self.embed_stats_change(state['stats_change']),
            self.embed_status(state['status']),
            self.embed_hp_ratio(state['hp_ratio']),
        ], dim=-1)                                   # (B, T, 12, 11*feat)
        pokemon_flat = pokemon_emb.flatten(2)        # (B, T, 12*11*feat)

        move_emb = torch.cat([
            self.embed_id_move(move['id']),
            self.embed_move_type(move['type']),
            self.embed_d_class(move['d_class']),
            self.embed_t_class(move['t_class']),
            self.embed_power(move['power']),
            self.embed_priority(move['priority'].unsqueeze(-1)),
            self.embed_accuracy(move['accuracy'].unsqueeze(-1)),
        ], dim=-1)                                   # (B, T, 12, 4, 7*feat)
        move_flat = move_emb.flatten(2)              # (B, T, 12*4*7*feat)

        weather_emb = self.embed_current_weather(battlefield['current_weather'])
        speed_emb = self.embed_speed_modifier(battlefield['speed_modifier'])

        full_state = torch.cat([pokemon_flat, move_flat,
                                weather_emb, speed_emb], dim=-1)
        state_emb = self.state_proj(full_state)      # (B, T, d_model)

        # --- action 
        full_action = torch.cat([
            self.embed_player_user(action['player_user']),
            self.embed_slot_user(action['slot_user']),
            self.embed_player_target(action['player_target']),
            self.embed_slot_target(action['slot_target']),
            self.embed_mega(action['mega']),
            self.embed_move(action['move']),
        ], dim=-1)                                   # (B, T, 2, 6*feat)
        action_emb = self.action_proj(full_action.flatten(2))

        # --- reward and turn 
        reward_emb = self.embed_reward(reward)
        turn_emb = self.embed_turn(turn)

        state_emb = state_emb + turn_emb
        action_emb = action_emb + turn_emb
        reward_emb = reward_emb + turn_emb

        # interleaving (R_t, s_t, a_t): (B, 3T, d_model)
        final_inputs = torch.stack(
            [reward_emb, state_emb, action_emb],
            dim=2).reshape(batch_size, 3 * seq_len, self.d_model)

        stacked_padding_mask = (padding_mask.repeat_interleave(3, dim=1)
                                if padding_mask is not None else None)
        return final_inputs, stacked_padding_mask
