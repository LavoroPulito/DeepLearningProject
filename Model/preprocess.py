# ── Vectorized Numpy preprocessing + augmentation ──
"""Vectorized Numpy preprocessing of an .npz file (a game).

Replaces Python's triple loops in __getitem__: structured Numpy arrays
allow vector access to all fields (e.g., turns['pokemon']['atk']
has shape (T, 12)). Includes remapping of IDs to embedding indices and
calculation of the legal mask, so the Dataset only needs to convert to torch.

Everything is pure Numpy: testable without torch.
"""

import numpy as np  # type: ignore

from id_maps import POKE_LUT, MOVE_LUT, ABILITY_LUT, ITEM_LUT, remap
from LegalActionMask import ActionMasker

PREPROCESS_VERSION = 2  # bump to invalidate the cache


def preprocess_game(file_path, max_turn=49):
    """Returns a dict of numpy arrays already max_turned and remapped."""
    with np.load(file_path, allow_pickle=True) as loaded:
        turns = loaded['turns']

    T = min(len(turns), max_turn)
    turns = turns[:T]
    poke = turns['pokemon']            # (T, 12) structured
    field = turns['field']             # (T,)

    def pad(a, fill=0):
        """Set the first dimension to max_turn."""
        if a.shape[0] == max_turn:
            return np.ascontiguousarray(a)
        width = [(0, max_turn - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
        return np.pad(a, width, constant_values=fill)

    # --- state -----------------------------------------------------------
    state = {
        'id':      pad(remap(POKE_LUT, poke['poke_id'])),
        'player':  pad(poke['player'].astype(np.int64)),
        'type1':   pad(poke['type1'].astype(np.int64)),
        'type2':   pad(poke['type2'].astype(np.int64)),
        'ability': pad(remap(ABILITY_LUT, poke['ability'])),
        'item':    pad(remap(ITEM_LUT, poke['item'])),
        'slot':    pad(poke['slot'].astype(np.int64)),
        'stats': pad(np.stack([poke['hp_base'], poke['atk'], poke['def_'],
                               poke['spa'], poke['spd'], poke['spe']],
                              axis=-1).astype(np.float32) / 255.0),
        'stats_change': pad(np.stack([poke['atk_c'], poke['def_c'],
                                      poke['spa_c'], poke['spd_c'],
                                      poke['spe_c']],
                                     axis=-1).astype(np.float32) / 6.0),
        'status': pad(((poke['status_mask'][..., None].astype(np.int64)
                        >> np.arange(5, -1, -1)) & 1).astype(np.float32)),
        'hp_ratio': pad(poke['hp_ratio'].astype(np.float32)[..., None]),
    }

    # --- mosse (stack dei 4 slot mossa) ------------------------------------
    mv = [poke[f'move{m}'] for m in range(4)]          # 4 x (T, 12)
    def mstack(f): return np.stack([m[f] for m in mv], axis=-1)  # (T, 12, 4)
    raw_move_id = mstack('id')
    move = {
        'id':      pad(remap(MOVE_LUT, raw_move_id)),
        'd_class': pad(mstack('d_class').astype(np.int64)),
        't_class': pad(mstack('t_class').astype(np.int64)),
        'type':    pad(mstack('type').astype(np.int64)),
        'power':   pad(mstack('power').astype(np.float32)[..., None] / 255.0),
        'priority': pad(mstack('priority').astype(np.float32) / 8.0),
        'accuracy': pad(mstack('accuracy').astype(np.float32) / 100.0),
    }

    # --- field -------------------------------------------------------------
    battlefield = {
        'current_weather': pad(field['weather'].astype(np.int64)),
        'speed_modifier': pad(((field['speed_mask'][..., None].astype(np.int64)
                                >> np.arange(2, -1, -1)) & 1).astype(np.float32)),
    }

    # --- action (stack action0/action1) -------------------------------------
    acts = [turns['action0'], turns['action1']]
    def astack(f): return np.stack([a[f].astype(np.int64) for a in acts], axis=-1)
    action = {
        'player_user':   pad(astack('usr_pl')),
        'slot_user':     pad(astack('usr_slot')),
        'player_target': pad(astack('trg_pl')),
        'slot_target':   pad(astack('trg_slot')),
        'mega':          pad(astack('mega')),
        'move':          pad(astack('move')),
    }

    # flat index (T, 2) in the 360 action space - same formula as the masker
    target_flat = ActionMasker.flat_batch(
        action['slot_user'], action['player_target'],
        action['slot_target'], action['mega'], action['move'])

    # --- reward and padding ----------------------------------------------------
    # Decision Transformer convention: return-to-go, i.e., the final outcome
    # replicated on every valid round (for inference, it is conditioned with 0 = win).
    # With the reward only on the last round, the model could not condition
    # the actions on the desired outcome.
    reward = np.zeros(max_turn, dtype=np.int64)
    reward[:T] = max(0, int(field['winner'][-1])) #be careful: winner contains the id of the winner indeed. reward == 0 means player 0 (the emulated) has win
    padding_mask = np.zeros(max_turn, dtype=np.int64)
    padding_mask[:T] = 1

    # --- legal mask -------------------------------------------------------
    # mega available on turn t if no previous action has mega evolved
    mega_any = (action['mega'][:, 0] | action['mega'][:, 1]).astype(bool)
    mega_used_before = np.concatenate([[False], np.cumsum(mega_any)[:-1] > 0])

    masker = ActionMasker()
    legal = np.ones((max_turn, masker.total_actions), dtype=bool)
    for t in range(T):
        st = {'player': state['player'][t], 'slot': state['slot'][t],
              'hp_ratio': state['hp_ratio'][t]}
        legal[t] = masker.get_valid_action_mask(
            st, {'id': raw_move_id[t]}, not mega_used_before[t])
        # safety net: the action actually played is always legal
        for a in range(2):
            idx = target_flat[t, a]
            if 0 <= idx < masker.total_actions:
                legal[t, idx] = True

    return {
        'state': state,
        'move': move,
        'battlefield': battlefield,
        'action': action,
        'reward': reward,
        'turn': np.arange(max_turn, dtype=np.int64),
        'padding_mask': padding_mask,
        'target_flat': target_flat,
        'legal_action_mask': legal,
    }


def augment_game(g, rng, permute_rows=True, permute_moves=True):
    """Data augmentation. Returns a COPY (does not change the Dataset cache).

    1. Permutes the order of the 4 moves of each Pokémon (one permutation per
    mon, constant across turns). The `move` index of the action is just the
    position in the moveset: permuting it teaches invariance. The
    move index of the target/input action and the
    `move` columns of the legal mask are consistently remapped (with the permutation of the
    mon that acts in that turn/slot).
    2. Permutes the 12 Pokémon rows in the status token: the position in the
    list is arbitrary (the information is in `player` and `slot`), so
    the target and mask do not change.
    """

    out = {}
    for k, v in g.items():
        out[k] = ({kk: vv.copy() for kk, vv in v.items()}
                  if isinstance(v, dict) else v.copy())

    T = int(out['padding_mask'].sum())
    player = g['state']['player']   # Original references
    slot = g['state']['slot']

    if permute_moves:
        perms = np.stack([rng.permutation(4) for _ in range(12)])  # (12, 4)
        inv = np.argsort(perms, axis=1)      # inv[i, old] = new index

        # permute move arrays: new[..., i, k] = old[..., i, perms[i, k]]
        for k, v in out['move'].items():
            idx = perms[None, :, :]
            if v.ndim == 4:                  # es. power (T, 12, 4, 1)
                idx = idx[..., None]
            out['move'][k] = np.take_along_axis(v, np.broadcast_to(idx, v.shape),
                                                axis=2)

        # remaps the action's move index (input and target)
        for t in range(T):
            for a in range(2):
                mv = int(out['action']['move'][t, a])
                s = int(out['action']['slot_user'][t, a])
                if mv < 4 and s in (1, 2):
                    rows = np.where((player[t] == 0) & (slot[t] == s))[0]
                    if rows.size:
                        out['action']['move'][t, a] = inv[int(rows[0]), mv]

        # remap the `move` columns of the legal form
        legal = out['legal_action_mask'][:T].reshape(T, 3, 2, 5, 2, 6)
        for t in range(T):
            for s in (1, 2):
                rows = np.where((player[t] == 0) & (slot[t] == s))[0]
                if rows.size:
                    p = perms[int(rows[0])]
                    legal[t, s, ..., :4] = np.take(legal[t, s], p, axis=-1)
        out['legal_action_mask'][:T] = legal.reshape(T, -1)

        # recalculate the flat target with the updated move index
        a = out['action']
        out['target_flat'] = ActionMasker.flat_batch(
            a['slot_user'], a['player_target'], a['slot_target'],
            a['mega'], a['move'])

    if permute_rows:
        rp = rng.permutation(12)
        for k in out['state']:
            out['state'][k] = out['state'][k][:, rp]
        for k in out['move']:
            out['move'][k] = out['move'][k][:, rp]

    return out
