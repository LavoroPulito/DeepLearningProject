"""Smoke test: verifica in ~1 minuto (CPU) che la pipeline giri e impari.

    source venv/bin/activate && python3 Model/smoke_test.py

Controlla: shapes, assenza di NaN (anche con padding), target sempre legale,
e che 60 step di overfit su 8 partite facciano crollare la loss e salire
l'accuracy (se il modello puo' memorizzare 8 partite, la pipeline e' sana).
"""
import glob
import sys
import time
from pathlib import Path

import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import DecisionTransformer

DATA = Path(__file__).resolve().parent.parent / 'npz' / 'reg_m-B'

def main():
    torch.manual_seed(0)
    files = sorted(glob.glob(str(DATA / '*.npz')))[:8]
    assert files, f'nessun npz in {DATA}'
    ds = PokemonVGCDataset(files, max_turn=49)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(dl))

    # 1. target sempre legale
    assert batch['legal_action_mask'].gather(-1, batch['target_flat']).all(), \
        'target illegale!'
    print('[ok] target sempre legale')

    model = DecisionTransformer(action_dim=360, d_model=64, n_heads=4,
                                depth=2, max_turn=49)
    print(f'[ok] modello: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parametri')

    # 2. forward: shapes e niente NaN sulle posizioni valide
    lp = model(batch['state'], batch['move'], batch['battlefield'],
               batch['action'], batch['reward'], batch['turn'],
               batch['padding_mask'],
               legal_action_mask=batch['legal_action_mask'])
    assert lp.shape == (8, 49, 2, 360), lp.shape
    valid = batch['padding_mask'].bool()
    assert torch.isfinite(lp[valid].max()), 'NaN/inf nei log-prob validi'
    print(f'[ok] forward: {tuple(lp.shape)}, log-prob finiti sui turni validi')

    # 3. overfit di 60 step su 8 partite
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = torch.nn.NLLLoss(reduction='none')
    t0 = time.time()
    first = last = acc = None
    for step in range(60):
        lp = model(batch['state'], batch['move'], batch['battlefield'],
                   batch['action'], batch['reward'], batch['turn'],
                   batch['padding_mask'],
                   legal_action_mask=batch['legal_action_mask'])
        le = crit(lp.reshape(-1, lp.size(-1)),
                  batch['target_flat'].reshape(-1)).view_as(batch['target_flat'])
        m = batch['padding_mask'].unsqueeze(-1).expand_as(le).float()
        loss = (le * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc = (((lp.argmax(-1) == batch['target_flat']).float() * m).sum()
               / m.sum()).item()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0:
            print(f'  step {step:2d}: loss {last:.3f}  acc {acc:.3f}')
    print(f'[ok] 60 step in {time.time()-t0:.0f}s: loss {first:.3f} -> {last:.3f}, acc {acc:.3f}')
    assert last < first * 0.5, 'la loss non scende abbastanza: qualcosa non va'
    print('\nTUTTO OK: la pipeline funziona e il modello impara.')

if __name__ == '__main__':
    main()
