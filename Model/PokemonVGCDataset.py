import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

class PokemonVGCDataset(Dataset):
    def __init__(self, file_paths, max_turn=49):
        self.file_paths = file_paths
        self.max_turn = max_turn

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Caricamento del file .npz o .npy
        file_path = self.file_paths[idx]
        
        # Apriamo il file e gestiamo l'estrazione di 'turns' se è un archivio .npz
        with np.load(file_path, allow_pickle=True) as loaded_file:
            data = loaded_file['turns'] if 'turns' in loaded_file else loaded_file

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
            'priority': torch.zeros((self.max_turn, 12, 4), dtype=torch.float32), # Vedi nota sotto
            'accuracy': torch.zeros((self.max_turn, 12, 4), dtype=torch.float32) # Vedi nota sotto
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
            battlefield['current_weather'][t] = int(turn_data['field']['weather'])
            battlefield['speed_modifier'][t] = torch.tensor((turn_data['field']['speed_mask'] >> np.arange(2, -1, -1)) & 1, dtype=torch.float32)

            # NOTA: speed_mask in numpy è uno scalare, ma l'embedding vuole 3 feature continue.
            # Dovrete scompattare speed_mask o modificare l'Embedding.
            
            # Popolamento Azioni (Stacking action0 e action1)
            for a_idx, act_key in enumerate(['action0', 'action1']):
                act_data = turn_data[act_key]
                action['player_user'][t, a_idx] = int(act_data['usr_pl'])
                action['slot_user'][t, a_idx] = int(act_data['usr_slot'])
                action['player_target'][t, a_idx] = int(act_data['trg_pl'])
                action['slot_target'][t, a_idx] = int(act_data['trg_slot'])
                action['mega'][t, a_idx] = int(act_data['mega'])
                action['move'][t, a_idx] = int(act_data['move'])

            # Popolamento Pokemon e Mosse
            for p in range(12):
                p_data = poke_data[p]
                
                # Features discrete
                state['id'][t, p] = int(p_data['poke_id'])
                state['type'][t, p] = int(p_data['type1']) # Usiamo type1 per compatibilità con l'embedding
                state['ability'][t, p] = int(p_data['ability'])
                state['item'][t, p] = int(p_data['item'])
                state['slot'][t, p] = int(p_data['slot'])
                
                # Features continue (Raggruppiamo i campi scalari in array)
                state['stats'][t, p] = torch.tensor([p_data['hp_base'], p_data['atk'], p_data['def_'], p_data['spa'], p_data['spd'], p_data['spe']], dtype=torch.float32)
                state['stats_change'][t, p] = torch.tensor([p_data['atk_c'], p_data['def_c'], p_data['spa_c'], p_data['spd_c'], p_data['spe_c']], dtype=torch.float32)
                state['hp_ratio'][t, p, 0] = float(p_data['hp_ratio'])
                state['status'][t, p] = torch.tensor((p_data['status_mask'] >> np.arange(5, -1, -1)) & 1, dtype=torch.float32)
                

                # Estrazione delle 4 mosse (Stacking move0, move1, move2, move3)
                for m_idx, move_key in enumerate(['move0', 'move1', 'move2', 'move3']):
                    m_data = p_data[move_key]
                    move['id'][t, p, m_idx] = int(m_data['id'])
                    move['d_class'][t, p, m_idx] = int(m_data['d_class'])
                    move['t_class'][t, p, m_idx] = int(m_data['t_class'])
                    move['power'][t, p, m_idx, 0] = float(m_data['power'])
                    move['priority'][t, p, m_idx] = float(m_data['priority'])
                    move['accuracy'][t, p, m_idx] = float(m_data['accuracy'])
                    

            # Esempio di gestione Reward (se winner è nel campo, assegnare 1 al turno finale o simile)
            if t == num_turns - 1: 
                 reward[t] = int(turn_data['field']['winner'])

        # Creazione del target_actions per la Loss (usando gli ID delle azioni scelte)
        target_actions = {k: v.clone() for k, v in action.items()} 

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

if __name__ == '__main__':

    # Definiamo direttamente il percorso al file .npz
    test_file = '../npz/reg_m-A/gen9championsvgc2026regma-2584198395.npz'

    # Passiamo una lista di percorsi file (lo mettiamo 2 volte per simulare un batch size di 2)
    dataset = PokemonVGCDataset(file_paths=[test_file, test_file], max_turn=49)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print("\n--- Inizio Estrazione Batch ---")
    # 3. Iteriamo e controlliamo i tensori
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Dimensioni 'state->id': {batch['state']['id'].shape} (Atteso: [2, 49, 12])")
        print(f"Dimensioni 'move->power': {batch['move']['power'].shape} (Atteso: [2, 49, 12, 4, 1])")
        print(f"Dimensioni 'action->move': {batch['action']['move'].shape} (Atteso: [2, 49, 2])")
        print(f"Padding mask per la prima partita (10 turni): {batch['padding_mask'][0].sum().item()} turni attivi")
        print(f"Padding mask per la seconda partita (20 turni): {batch['padding_mask'][1].sum().item()} turni attivi")
        
        # Stampiamo i dati binari di status e speed_modifier per il primo step
        for t in range(9):
            print(f"Esempio maschera status de-binarizzata (Partita 1, Turno 0, Pkm 0): {batch['battlefield']['speed_modifier'][0, t]}")
        break # Ci basta testare un batch