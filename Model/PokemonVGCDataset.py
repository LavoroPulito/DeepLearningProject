import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PokemonVGCDataset(Dataset):
    def __init__(self, file_paths, max_turn=48):
        self.file_paths = file_paths
        self.max_turn = max_turn

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Caricamento del file .npy (shape: (N_turni,))
        data = np.load(self.file_paths[idx])
        num_turns = min(len(data), self.max_turn) # Tronchiamo se supera max_turn
        
        # Inizializziamo i tensori vuoti con il padding (es. zeri)
        # Dimensioni target: (max_turn, 12, ...) per state/move, (max_turn, 2, ...) per action
        
        # --- PREPARAZIONE DIZIONARI VUOTI ---
        state = {
            'id': torch.zeros((self.max_turn, 12), dtype=torch.long),
            'type': torch.zeros((self.max_turn, 12), dtype=torch.long),
            'ability': torch.zeros((self.max_turn, 12), dtype=torch.long),
            'item': torch.zeros((self.max_turn, 12), dtype=torch.long),
            'slot': torch.zeros((self.max_turn, 12), dtype=torch.long),
            'stats': torch.zeros((self.max_turn, 12, 6), dtype=torch.float32),
            'stats_change': torch.zeros((self.max_turn, 12, 5), dtype=torch.float32),
            'status': torch.zeros((self.max_turn, 12, 6), dtype=torch.float32), # Vedi nota sotto
            'hp_ratio': torch.zeros((self.max_turn, 12, 1), dtype=torch.float32)
        }
        
        move = {
            'id': torch.zeros((self.max_turn, 12, 4), dtype=torch.long),
            'd_class': torch.zeros((self.max_turn, 12, 4), dtype=torch.long),
            't_class': torch.zeros((self.max_turn, 12, 4), dtype=torch.long),
            'power': torch.zeros((self.max_turn, 12, 4, 1), dtype=torch.float32),
            'priority': torch.zeros((self.max_turn, 12, 4, 7), dtype=torch.float32), # Vedi nota sotto
            'accuracy': torch.zeros((self.max_turn, 12, 4, 100), dtype=torch.float32) # Vedi nota sotto
        }
        
        battlefield = {
            'current_weather': torch.zeros((self.max_turn,), dtype=torch.long),
            'speed_modifier': torch.zeros((self.max_turn, 3), dtype=torch.float32) # Vedi nota sotto
        }
        
        action = {
            'player_user': torch.zeros((self.max_turn, 2), dtype=torch.long),
            'slot_user': torch.zeros((self.max_turn, 2), dtype=torch.long),
            'player_target': torch.zeros((self.max_turn, 2), dtype=torch.long),
            'slot_target': torch.zeros((self.max_turn, 2), dtype=torch.long),
            'mega': torch.zeros((self.max_turn, 2), dtype=torch.long),
            'move': torch.zeros((self.max_turn, 2), dtype=torch.long)
        }

        reward = torch.zeros((self.max_turn,), dtype=torch.long)
        turn_tensor = torch.arange(self.max_turn, dtype=torch.long)
        
        # Maschera per i turni validi
        padding_mask = torch.zeros((self.max_turn,), dtype=torch.long)
        padding_mask[:num_turns] = 1

        # 2. ESTRAZIONE DATI DAL NUMPY E POPOLAMENTO TENSORI
        for t in range(num_turns):
            turn_data = data[t]
            poke_data = turn_data['pokemon'] # Array di 12 pokemon
            
            # Popolamento Campo (Battlefield)
            battlefield['current_weather'][t] = turn_data['field']['weather']
            # NOTA: speed_mask in numpy è uno scalare, ma l'embedding vuole 3 feature continue.
            # Dovrete scompattare speed_mask o modificare l'Embedding.
            
            # Popolamento Azioni (Stacking action0 e action1)
            for a_idx, act_key in enumerate(['action0', 'action1']):
                act_data = turn_data[act_key]
                action['player_user'][t, a_idx] = act_data['usr_pl']
                action['slot_user'][t, a_idx] = act_data['usr_slot']
                action['player_target'][t, a_idx] = act_data['trg_pl']
                action['slot_target'][t, a_idx] = act_data['trg_slot']
                action['mega'][t, a_idx] = act_data['mega']
                action['move'][t, a_idx] = act_data['move']

            # Popolamento Pokemon e Mosse
            for p in range(12):
                p_data = poke_data[p]
                
                # Features discrete
                state['id'][t, p] = p_data['poke_id']
                state['type'][t, p] = p_data['type1'] # Usiamo type1 per compatibilità con l'embedding
                state['ability'][t, p] = p_data['ability']
                state['item'][t, p] = p_data['item']
                state['slot'][t, p] = p_data['slot']
                
                # Features continue (Raggruppiamo i campi scalari in array)
                state['stats'][t, p] = torch.tensor([p_data['hp_base'], p_data['atk'], p_data['def_'], p_data['spa'], p_data['spd'], p_data['spe']], dtype=torch.float32)
                state['stats_change'][t, p] = torch.tensor([p_data['atk_c'], p_data['def_c'], p_data['spa_c'], p_data['spd_c'], p_data['spe_c']], dtype=torch.float32)
                state['hp_ratio'][t, p, 0] = p_data['hp_ratio']
                
                # NOTA: status_mask è scalare, ma l'embedding si aspetta 6 feature. 
                # Dovrete spacchettare i bit della maschera in un array di 6 elementi.

                # Estrazione delle 4 mosse (Stacking move0, move1, move2, move3)
                for m_idx, move_key in enumerate(['move0', 'move1', 'move2', 'move3']):
                    m_data = p_data[move_key]
                    move['id'][t, p, m_idx] = m_data['id']
                    move['d_class'][t, p, m_idx] = m_data['d_class']
                    move['t_class'][t, p, m_idx] = m_data['t_class']
                    move['power'][t, p, m_idx, 0] = m_data['power']
                    # NOTA: priority e accuracy nell'embedding richiedono feature espansive (7 e 100),
                    # dovrete applicare un one-hot encoding o adattare il layer Linear.

            # Esempio di gestione Reward (se winner è nel campo, assegnare 1 al turno finale o simile)
            if t == num_turns - 1: 
                 reward[t] = turn_data['field']['winner']

        # Creazione del target_actions per la Loss (usando gli ID delle azioni scelte)
        target_actions = action['move'].clone() 

        return {
            'state': state,
            'move': move,
            'battlefield': battlefield,
            'action': action,
            'reward': reward,
            'turn': turn_tensor,
            'padding_mask': padding_mask,
            'target_actions': target_actions
        }
