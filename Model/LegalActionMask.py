# ── Legal Action Mask ───────────────────────────
"""Legal Action Mask.

Action Space (flat, 360):
dims = (s_user: 3, p_target: 2, s_target: 5, mega: 2, move: 6)
flat = s_user*120 + p_target*60 + s_target*12 + mega*6 + move

Replay conventions (see cleanstrings.py, validate with validate_mask.py):
- pass -> (s_user=0, trg=(0,0), mega=0, move=5)
- switch (move=4) -> trg_slot = slot that the ENTERING player had before: 
    0 = never entered the game, 3/4 = bench
- move m<4 -> trg_pl in {0,1}, trg_slot in {1,2}
- move=5 with s_user>0 = move not recognized (Struggle, etc.)
- p_user is always 0 in the data, so it's not a dimension.

Known limitations (this is why the mask is permissive, never strict):
- the snapshot is at the beginning of the turn: it doesn't see intra-turn dynamics
(mon acting and then dying, intra-turn benching after a switch);
- moves are revealed on first use: a new move has id=0
at the first free index of the moveset;
- the targets of self/spread moves are noisy (~5-7%).

Anyway the dataset forces mask[true_action] = True as a safety net:
a -inf on the true action would make the loss infinite.

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
        return s_user * 120 + p_target * 60 + s_target * 12 + mega * 6 + move

    def get_valid_action_mask(self, state, move, mega_available):
        """Boolean mask [360] for one turn.

                state: numpy dict with 'player', 'slot', 'hp_ratio' (12 mon)
                move: numpy dict with 'id' shape (12, 4)
                mega_available: True if the player has not yet mega-evolved
                    (from action history, calculated in the Dataset).
        """
        mask = np.zeros(self.total_actions, dtype=bool)

        player = np.asarray(state['player']).reshape(12)
        slot = np.asarray(state['slot']).reshape(12)
        hp = np.asarray(state['hp_ratio']).reshape(12)
        move_id = np.asarray(move['id']).reshape(12, 4)
        own = player == 0
        own_alive = own & (hp > 0)
        any_own_active_alive = bool(np.any(own_alive & ((slot == 1) | (slot == 2))))

        # --- pass: always available (empty slot / replacement / end of game)
        mask[self.flat(0, 0, 0, 0, 5)] = True

        megas = (0, 1) if mega_available else (0,)

        for s in (1, 2):
            # --- switch (move=4): allowed even on dead slot (replacement)
            for bench in (3, 4):
                if np.any(own_alive & (slot == bench)) or any_own_active_alive:
                    mask[self.flat(s, 0, bench, 0, 4)] = True
            # towards a mon never taken to the field (in bring-4 they always exist)
            if np.any(own & (slot == 0)):
                mask[self.flat(s, 0, 0, 0, 4)] = True

            # --- moves: check presence of a mon in the user slot  
            idx = np.where(own & (slot == s))[0]
            if idx.size == 0:
                continue
            i = int(idx[0])
            ids = move_id[i]
            first_free = next((j for j in range(4) if ids[j] == 0), None)
            for m in range(4):
                # id==0 legal only on the first slot 
                if ids[m] == 0 and m != first_free:
                    continue
                for trg_pl in (0, 1):
                    for trg_sl in (1, 2):
                        for mg in megas:
                            mask[self.flat(s, trg_pl, trg_sl, mg, m)] = True
            for trg_pl in (0, 1):
                for trg_sl in (1, 2):
                    mask[self.flat(s, trg_pl, trg_sl, 0, 5)] = True

        return mask
