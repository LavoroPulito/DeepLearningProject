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
        # Le tue dimensioni: 2x2x2x4x2x6
        self.dims = (2, 2, 2, 4, 2, 6)
        self.total_actions = np.prod(self.dims) # 192

    def get_flat_index(self, p_user, s_user, p_target, s_target, mega, move):
        """Converte le 6 componenti in un singolo indice da 0 a 191"""
        return np.ravel_multi_index(
            (p_user, s_user, p_target, s_target, mega, move), 
            self.dims
        )

    def get_valid_action_mask(self, state):
        """
        Analizza lo stato corrente e restituisce una maschera booleana [192]
        dove True = Azione consentita, False = Azione illegale.
        """
        # Partiamo assumendo che nessuna azione sia valida
        mask = np.zeros(self.total_actions, dtype=bool)

        p_user = 0 # Assumiamo che 0 sia il giocatore corrente (dimensione 1)

        # Controlliamo le azioni per ciascuno dei tuoi 2 Pokémon attivi
        for s_user in range(2): 
            
            # Se il Pokémon in questo slot è esausto, non può fare nulla.
            # (Dovrai implementare tu la logica di `is_pokemon_alive(state, s_user)`)
            if not self.is_pokemon_alive(state, s_user):
                continue

            # ----------------------------------------------------
            # 1. CONTROLLO SWITCH (Mosse 4 e 5)
            # ----------------------------------------------------
            # Puoi fare switch se non sei trappolato.
            if self.can_switch(state, s_user):
                for switch_idx in [4, 5]: # Le due opzioni di switch in panchina
                    if self.is_bench_pokemon_available(state, switch_idx):
                        # Per gli switch il target e la mega non hanno senso. 
                        # Usiamo valori "dummy" standard (es. tutti zeri) per queste dimensioni.
                        idx = self.get_flat_index(p_user, s_user, 0, 0, 0, switch_idx)
                        mask[idx] = True

            # ----------------------------------------------------
            # 2. CONTROLLO MOSSE D'ATTACCO (Mosse da 0 a 3)
            # ----------------------------------------------------
            can_mega = self.can_mega_evolve(state, s_user)
            mega_options = [0, 1] if can_mega else [0] # 1=Mega sì, 0=Mega no

            for move_idx in range(4):
                if self.can_use_move(state, s_user, move_idx):
                    
                    # Recupera i bersagli validi per QUESTA specifica mossa 
                    # (es. Terremoto colpisce tutti, Protezione colpisce se stessi)
                    valid_targets = self.get_valid_targets_for_move(state, s_user, move_idx)
                    
                    for (p_target, s_target) in valid_targets:
                        for mega in mega_options:
                            idx = self.get_flat_index(p_user, s_user, p_target, s_target, mega, move_idx)
                            mask[idx] = True

        return mask

    # --- METODI DUMMY (Da implementare con la logica del tuo simulatore) ---
    def is_pokemon_alive(self, state, slot): 
        return next((p for p in mons if p.player == player and p.slot == slot))
        return True
# |-|-|-|-|-|-|
    def can_switch(self, state, slot): return True
    def is_bench_pokemon_available(self, state, bench_idx): return True
    def can_mega_evolve(self, state, slot): return False
    def can_use_move(self, state, slot, move_idx): return True # Controlla PP, Disable, Taunt...
    def get_valid_targets_for_move(self, state, slot, move_idx):
        # Restituisce una lista di tuple (p_target, s_target)
        # Esempio: se è una mossa singola verso il nemico potresti restituire [(1, 0), (1, 1)]
        return [(1, 0), (1, 1)]