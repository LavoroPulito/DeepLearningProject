import sys
from pathlib import Path

# Ottieni il percorso assoluto della cartella 'src'
# __file__ è il file corrente, parent è la cartella corrente, il secondo parent è '..'
src_path = Path(__file__).resolve().parent.parent / 'src'

# Aggiungi il percorso al sys.path
sys.path.append(str(src_path))

# Ora puoi importare normalmente il tuo file (senza l'estensione .py)
import getter

import numpy as np # type: ignore

class ActionMasker:
    def __init__(self):
        # Le tue dimensioni: 2x2x2x5x2x6
        self.dims = (2, 2, 2, 5, 2, 6)
        self.total_actions = np.prod(self.dims) # 480

    def get_flat_index(self, p_user, s_user, p_target, s_target, mega, move):
        """Converte le 6 componenti in un singolo indice da 0 a 191"""
        return np.ravel_multi_index(
            (p_user, s_user, p_target, s_target, mega, move), 
            self.dims
        )

    def get_valid_action_mask(self, state, move, first_state):
        """
        Analizza lo stato corrente e restituisce una maschera booleana [192]
        dove True = Azione consentita, False = Azione illegale.
        """
        mega_disp = False
        if state['player'][0] == 0:
            current_poke_id = state['id'][:6]
            first_poke_id = first_state['id'][:6]
        else:
            current_poke_id = state['id'][6:12]
            first_poke_id = first_state['id'][6:12]
        
        if np.array_equal(current_poke_id, first_poke_id):
            mega_disp = True
        

        

        # Partiamo assumendo che nessuna azione sia valida
        mask = np.zeros(self.total_actions, dtype=bool)
        if np.array_equal(state['id'], np.zeros_like(state['id'])):
            mask = np.ones(self.total_actions, dtype=bool)

        # p_user = 0  Assumiamo che 0 sia il giocatore corrente (dimensione 1)
        mask[self.get_flat_index(0, 0, 0, 0, 0, 5)] = True
        mask[self.get_flat_index(0, 1, 0, 1, 0, 5)] = True

        # Controlliamo le azioni per ciascuno dei tuoi 2 Pokémon attivi
        for s_user in [1,2]: 
            
            for i in range(12):
                #se esiste vivo in 3 -> sw in 3 disp
                if state['slot'][i] == 3 and state['hp_ratio'][i] != 0:
                    mask[self.get_flat_index(0, s_user-1, 0, 3, 0, 4)] = True
                #se esiste vivo in 4 -> sw in 4 disp
                if state['slot'][i] == 4 and state['hp_ratio'][i] != 0:
                    mask[self.get_flat_index(0, s_user-1, 0, 4, 0, 4)] = True
                
            if not self.present_in(state,4):
                mask[self.get_flat_index(0, s_user-1, 0, 0, 0, 4)] = True
                
            if self.is_pokemon_alive(state, s_user):
                for mve in [0,1,2,3]:
                    for trg_pl in [0,1]:
                        for trg_sl in [1,2]:
                            for i in range(12):
                                if state['slot'][i] == s_user and state['player'][i] == 0 and move['id'][i][mve] != 0:
                                    if move['t_class'][i][mve] in [4,13,5] and trg_pl == 0: 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True
                                    elif move['t_class'][i][mve] in [3,11] and trg_pl == 1: 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True
                                    elif move['t_class'][i][mve] in [12,14,16]: 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True
                                    elif move['t_class'][i][mve] in [9, 1, 2, 8, 10] and (trg_pl != 0 or trg_sl != s_user): 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True
                                    elif move['t_class'][i][mve] in [3,15] and trg_pl == 0 and trg_sl != s_user: 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True
                                    elif move['t_class'][i][mve] in [7] and trg_pl == 0 and trg_sl == s_user: 
                                        mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 0, mve)] = True
                                        if mega_disp and state['item'][i] == 2177:  
                                            mask[self.get_flat_index(0, s_user-1, trg_pl, trg_sl, 1, mve)] = True

        return mask

    # --- METODI DUMMY (Da implementare con la logica del tuo simulatore) ---
    def is_pokemon_alive(self, state, slot):
        #poke slot = 0 -> slot sconosciuto 
        for i in range(12):
            if state['slot'][i] == slot and state['player'][i] == 0:
                if state['hp_ratio'][i][0] == 0.0:
                    return False
                else:
                    return True
        return False

    def present_in(self, state, slot):
        for i in range (12):
            if state['slot'][i] == slot: 
                return True
        return False 
