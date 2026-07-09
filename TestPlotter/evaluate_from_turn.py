"""Valutazione del modello a partire da un certo turno in poi.

Il modello vede sempre l'intera storia della partita come contesto (e'
autoregressivo); a cambiare e' solo su QUALI turni si calcolano le metriche:
loss e accuracy vengono ristrette ai turni >= --from-turn. In piu' stampa
sempre il breakdown per fasce di turno, utile per capire in quale fase di
gioco il modello e' piu' o meno bravo (l'early game e' piu' stereotipato,
il late game piu' situazionale).

Uso (nel venv):
    python3 Model/evaluate_from_turn.py pesi.pth --data npz/reg_m-B --from-turn 10
    python3 Model/evaluate_from_turn.py pesi.pth --games 200 --buckets 0,3,6,10,15,25

L'architettura e' dedotta dai pesi (solo --heads se diverso da 8).
Nota: usa gli ULTIMI file in ordine alfabetico come test di default; passa
--all-files se vuoi valutare su tutto (attento a non valutare sul train!).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import DecisionTransformer
import random

def load_model(weights_path, n_heads):
    sd = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    d_model = sd['token_embedding.state_proj.weight'].shape[0]
    depth = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('tblocks.'))
    max_turn = sd['token_embedding.embed_turn.weight'].shape[0]
    action_dim = sd['predict_action.weight'].shape[0] // 2
    print(f'Architettura dedotta: d_model={d_model} depth={depth} '
          f'max_turn={max_turn} action_dim={action_dim}')
    model = DecisionTransformer(action_dim=action_dim, d_model=d_model,
                                n_heads=n_heads, depth=depth,
                                max_turn=max_turn, dropout=0.0)
    model.load_state_dict(sd)
    model.eval()
    return model, max_turn


@torch.no_grad()
def collect(model, dl, use_legal_mask=True):
    """Ritorna per ogni azione valida: turno, loss, corretto (1/0)."""
    criterion = nn.NLLLoss(reduction='none')
    turns, losses, corrects = [], [], []
    for batch in dl:
        legal = batch['legal_action_mask'] if use_legal_mask else None
        log_probs = model(batch['state'], batch['move'], batch['battlefield'],
                          batch['action'], batch['reward'], batch['turn'],
                          batch['padding_mask'], legal_action_mask=legal)
        target = batch['target_flat']
        loss_el = criterion(log_probs.reshape(-1, log_probs.size(-1)),
                            target.reshape(-1)).view_as(target)   # (B, T, 2)
        correct = (log_probs.argmax(-1) == target)                # (B, T, 2)
        valid = batch['padding_mask'].bool().unsqueeze(-1).expand_as(target)
        turn2 = batch['turn'].unsqueeze(-1).expand_as(target)
        turns.append(turn2[valid].numpy())
        losses.append(loss_el[valid].numpy())
        corrects.append(correct[valid].numpy())
    return (np.concatenate(turns), np.concatenate(losses),
            np.concatenate(corrects))


def report(turns, losses, corrects, lo, hi=None, label=None):
    m = turns >= lo if hi is None else (turns >= lo) & (turns < hi)
    if m.sum() == 0:
        return
    label = label or (f'turni >= {lo}' if hi is None else f'turni {lo}-{hi - 1}')
    print(f'  {label:>14}: loss {losses[m].mean():.4f} | '
          f'acc {corrects[m].mean():.3f} | n azioni {int(m.sum())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('--data', default=str(Path(__file__).resolve().parent.parent
                                          / 'npz' / 'reg_m-B'))
    ap.add_argument('--from-turn', type=int, default=0)
    ap.add_argument('--games', type=int, default=200,
                    help='quante partite usare (le ultime in ordine alfabetico)')
    ap.add_argument('--all-files', action='store_true')
    ap.add_argument('--buckets', default='0,3,6,10,15,25',
                    help='estremi delle fasce di turno per il breakdown')
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--no-legal-mask', action='store_true')
    args = ap.parse_args()

    model, max_turn = load_model(args.weights, args.heads)

    files = sorted(glob.glob('.../*.npz'))
    random.Random(42).shuffle(files)
    files = files[len(files) - len(files)//10:]   # come nel notebook
    if not files:
        sys.exit(f'nessun npz in {args.data}')
    if not args.all_files:
        files = files[-args.games:]
    print(f'{len(files)} partite da {args.data}')

    ds = PokemonVGCDataset(files, max_turn=max_turn)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    turns, losses, corrects = collect(model, dl,
                                      use_legal_mask=not args.no_legal_mask)

    print(f'\nTotale ({int(len(turns))} azioni):')
    report(turns, losses, corrects, 0, label='tutti i turni')
    if args.from_turn > 0:
        report(turns, losses, corrects, args.from_turn)

    print('\nBreakdown per fascia di turno:')
    edges = [int(x) for x in args.buckets.split(',')]
    for lo, hi in zip(edges, edges[1:]):
        report(turns, losses, corrects, lo, hi)
    report(turns, losses, corrects, edges[-1])


if __name__ == '__main__':
    main()
