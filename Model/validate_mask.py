"""Valida la LegalActionMask contro le azioni realmente giocate nei replay.

Ogni azione presa in un replay e' per definizione legale: se la maschera la
marca illegale (o l'indice piatto non e' rappresentabile) c'e' un bug.

Uso:  python3 Model/validate_mask.py [n_files]
"""
import sys
import glob
from collections import Counter

import numpy as np

from LegalActionMask import ActionMasker


def flat_index(a):
    """Stessa codifica di TrainingLoop: s_user*120+p_trg*60+s_trg*12+mega*6+move."""
    return (int(a['usr_slot']) * 120
            + int(a['trg_pl']) * 60
            + int(a['trg_slot']) * 12
            + int(a['mega']) * 6
            + int(a['move']))


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(n_files)
    files = sorted(glob.glob('../npz/reg_m-B/*.npz'))[:n_files]
    masker = ActionMasker()

    total = 0
    out_of_range = Counter()
    illegal = Counter()
    ok = 0
    n_legali_media = []

    for f in files:
        with np.load(f, allow_pickle=True) as d:
            turns = d['turns']
        mega_used = False
        for t in turns:
            poke = t['pokemon']
            state = {
                'player':   np.array([int(p['player']) for p in poke]),
                'slot':     np.array([int(p['slot']) for p in poke]),
                'hp_ratio': np.array([float(p['hp_ratio']) for p in poke]),
            }
            move_np = {'id': np.array([[int(poke[i][f'move{m}']['id'])
                                        for m in range(4)] for i in range(12)])}
            mask = masker.get_valid_action_mask(state, move_np, not mega_used)
            n_legali_media.append(mask.sum())
            for ak in ('action0', 'action1'):
                a = t[ak]
                total += 1
                idx = flat_index(a)
                key = (int(a['move']), int(a['usr_slot']), int(a['trg_pl']),
                       int(a['trg_slot']), int(a['mega']))
                if idx < 0 or idx >= masker.total_actions:
                    out_of_range[key] += 1
                elif not mask[idx]:
                    illegal[key] += 1
                else:
                    ok += 1
                if int(a['mega']) == 1:
                    mega_used = True

    print(f'Azioni totali: {total}  |  legali secondo la maschera: {ok} '
          f'({100 * ok / total:.2f}%)')
    print(f'Azioni legali per turno (media): {np.mean(n_legali_media):.1f} / '
          f'{masker.total_actions}')
    if out_of_range:
        print(f'\nFUORI RANGE: {sum(out_of_range.values())}')
        for k, v in out_of_range.most_common(10):
            print(f'  (move,usr_slot,trg_pl,trg_slot,mega)={k}: {v}')
    if illegal:
        print(f'\nMarcate ILLEGALI: {sum(illegal.values())}')
        for k, v in illegal.most_common(20):
            print(f'  (move,usr_slot,trg_pl,trg_slot,mega)={k}: {v}')


if __name__ == '__main__':
    main()
