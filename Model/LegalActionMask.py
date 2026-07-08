# ── Maschera azioni legali ───────────────────────────
"""Maschera delle azioni legali.

Spazio azioni (piatto, 360):
    dims = (s_user: 3, p_target: 2, s_target: 5, mega: 2, move: 6)
    flat  = s_user*120 + p_target*60 + s_target*12 + mega*6 + move

Convenzioni dei replay (vedi cleanstrings.py, validate con validate_mask.py):
  - pass            -> (s_user=0, trg=(0,0), mega=0, move=5)
  - switch (move=4) -> trg_slot = slot che il mon ENTRANTE aveva prima:
                       0 = mai sceso in campo, 3/4 = panchina
  - mossa m<4       -> trg_pl in {0,1}, trg_slot in {1,2}
  - move=5 con s_user>0 = mossa non riconosciuta (Struggle ecc.)
  - p_user e' sempre 0 nei dati, quindi non e' una dimensione.

Limiti noti (per questo la maschera e' permissiva, mai severa):
  - lo snapshot e' a inizio turno: non vede le dinamiche intra-turno
    (mon che agisce e poi muore, benching intra-turno dopo uno switch);
  - le mosse vengono rivelate al primo uso: una mossa nuova ha id==0
    al primo indice libero del moveset;
  - i target di mosse self/spread sono rumorosi (~5-7%).
Il Dataset forza comunque mask[azione_vera] = True come rete di sicurezza:
un -inf sull'azione vera renderebbe la loss infinita.
"""
import numpy as np  # type: ignore


class ActionMasker:
    DIMS = (3, 2, 5, 2, 6)          # s_user, p_target, s_target, mega, move
    TOTAL = 360

    def __init__(self):
        self.dims = self.DIMS
        self.total_actions = self.TOTAL

    @staticmethod
    def flat(s_user, p_target, s_target, mega, move):
        return s_user * 120 + p_target * 60 + s_target * 12 + mega * 6 + move

    @staticmethod
    def flat_batch(s_user, p_target, s_target, mega, move):
        """Versione vettoriale/tensoriale (funziona con numpy e torch)."""
        return s_user * 120 + p_target * 60 + s_target * 12 + mega * 6 + move

    def get_valid_action_mask(self, state, move, mega_available):
        """Maschera booleana [360] per un turno.

        state: dict numpy con 'player', 'slot', 'hp_ratio' (12 mon)
        move:  dict numpy con 'id' shape (12, 4)
        mega_available: True se il giocatore non ha ancora megaevoluto
                        (dalla storia delle azioni, calcolata nel Dataset).
        """
        mask = np.zeros(self.total_actions, dtype=bool)

        player = np.asarray(state['player']).reshape(12)
        slot = np.asarray(state['slot']).reshape(12)
        hp = np.asarray(state['hp_ratio']).reshape(12)
        move_id = np.asarray(move['id']).reshape(12, 4)
        own = player == 0
        own_alive = own & (hp > 0)
        any_own_active_alive = bool(np.any(own_alive & ((slot == 1) | (slot == 2))))

        # --- pass: sempre disponibile (slot vuoto / rimpiazzo / fine gioco)
        mask[self.flat(0, 0, 0, 0, 5)] = True

        megas = (0, 1) if mega_available else (0,)

        for s in (1, 2):
            # --- switch (move=4): consentito anche a slot morto (rimpiazzo)
            # verso panchina: viva allo snapshot, oppure un attivo vivo
            # potrebbe finirci durante il turno (benching intra-turno)
            for bench in (3, 4):
                if np.any(own_alive & (slot == bench)) or any_own_active_alive:
                    mask[self.flat(s, 0, bench, 0, 4)] = True
            # verso un mon mai sceso in campo (in bring-4 esistono sempre)
            if np.any(own & (slot == 0)):
                mask[self.flat(s, 0, 0, 0, 4)] = True

            # --- mosse: serve un mon proprio nello slot s.
            # Non richiediamo hp>0: se e' morto allo snapshot ha agito e
            # poi e' caduto nello stesso turno (limite dello snapshot).
            idx = np.where(own & (slot == s))[0]
            if idx.size == 0:
                continue
            i = int(idx[0])
            ids = move_id[i]
            first_free = next((j for j in range(4) if ids[j] == 0), None)
            for m in range(4):
                # id==0 e' legale solo al primo indice libero:
                # e' li' che lo scraper registra una mossa al primo uso
                if ids[m] == 0 and m != first_free:
                    continue
                for trg_pl in (0, 1):
                    for trg_sl in (1, 2):
                        for mg in megas:
                            mask[self.flat(s, trg_pl, trg_sl, mg, m)] = True
            # move=5 con s_user>0: mossa non riconosciuta (Struggle ecc.)
            for trg_pl in (0, 1):
                for trg_sl in (1, 2):
                    mask[self.flat(s, trg_pl, trg_sl, 0, 5)] = True

        return mask
