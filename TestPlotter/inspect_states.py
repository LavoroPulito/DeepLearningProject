"""Ispezione delle rappresentazioni CONTESTUALI del modello.

Mentre inspect_embeddings.py guarda le tabelle statiche (i "pezzi"),
qui si guarda come il modello rappresenta le SITUAZIONI di gioco:
l'output del transformer sul token di stato (prima della testa), un vettore
d_model per ogni turno di ogni partita. Proiettato in 2D (PCA) e colorato:

  - per turno  -> si vede se il modello organizza le partite lungo una
                  "traiettoria" temporale (early/mid/late game)
  - per esito  -> si vede se vittorie e sconfitte si separano gia' nello
                  spazio degli stati (e da che turno in poi)

Uso (nel venv, servono torch e matplotlib):
    python3 Model/inspect_states.py pesi.pth --data npz/reg_m-B --games 60

Output: states_by_turn.png, states_by_outcome.png + statistiche a terminale.
L'architettura (d_model, depth, max_turn) viene dedotta dai pesi; solo
n_heads non e' deducibile (--heads, default 8, non cambia le shape).
NB: id_maps.py deve essere lo stesso usato per addestrare i pesi, altrimenti
le embedding table non combaciano.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np # type: ignore
import torch  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import DecisionTransformer
from torch.utils.data import DataLoader  # type: ignore


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
def encode_states(model, batch):
    """Replica il forward fino al token di stato, senza la testa."""
    x, mask = model.token_embedding(
        batch['state'], batch['move'], batch['battlefield'], batch['action'],
        batch['reward'], batch['turn'], batch['padding_mask'])
    for blk in model.tblocks:
        x = blk(x, padding_mask=mask)
    B = batch['padding_mask'].shape[0]
    return x.reshape(B, model.seq_length, 3, model.d_model)[:, :, 1]  # (B,T,d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights')
    ap.add_argument('--data', default=str(Path(__file__).resolve().parent.parent
                                          / 'npz' / 'reg_m-B'))
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--heads', type=int, default=12)
    args = ap.parse_args()

    model, max_turn = load_model(args.weights, args.heads)

    files = sorted(glob.glob(f'{args.data}/*.npz'))[:args.games]
    if not files:
        sys.exit(f'nessun npz in {args.data}')
    ds = PokemonVGCDataset(files, max_turn=max_turn)
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    reps, turns, outcomes = [], [], []
    for batch in dl:
        h = encode_states(model, batch)                    # (B, T, d)
        valid = batch['padding_mask'].bool()               # (B, T)
        reps.append(h[valid].numpy())
        turns.append(batch['turn'][valid].numpy())
        outcomes.append(batch['reward'][valid].numpy())    # esito costante
    H = np.concatenate(reps)          # (N, d)
    T = np.concatenate(turns)
    W = np.concatenate(outcomes)
    print(f'{H.shape[0]} stati da {len(files)} partite, dim {H.shape[1]}')

    # PCA
    Hc = H - H.mean(axis=0)
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    xy = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())
    print(f'varianza spiegata PC1/PC2: {var[0]:.0%} / {var[1]:.0%}')

    # statistiche testuali
    corr = np.corrcoef(xy[:, 0], T)[0, 1]
    print(f'correlazione PC1 ~ turno: {corr:+.2f}')
    mu0, mu1 = Hc[W == 0].mean(axis=0), Hc[W == 1].mean(axis=0)
    sep = np.linalg.norm(mu1 - mu0) / Hc.std(axis=0).mean()
    print(f'separazione win/loss (dist. tra centroidi / std media): {sep:.2f}')
    for lo, hi in ((0, 5), (5, 15), (15, max_turn)):
        m = (T >= lo) & (T < hi)
        if m.sum() < 10:
            continue
        d = (np.linalg.norm(Hc[m & (W == 1)].mean(0) - Hc[m & (W == 0)].mean(0))
             / Hc[m].std(axis=0).mean())
        print(f'  separazione win/loss nei turni {lo}-{hi}: {d:.2f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=T, s=6, cmap='viridis', alpha=0.7)
    fig.colorbar(sc, label='turno')
    ax.set_title('Rappresentazioni di stato — colorate per turno')
    fig.tight_layout()
    fig.savefig('states_by_turn.png', dpi=160)

    fig, ax = plt.subplots(figsize=(10, 8))
    for w, (col, lab) in enumerate([('tab:green', 'vittoria (R=0)'),
                                    ('tab:red', 'sconfitta (R=1)')]):
        # NB: convenzione dei dati: reward = indice del vincitore,
        # quindi 0 = vince il player 0 (il punto di vista dei replay)
        m = W == w
        ax.scatter(xy[m, 0], xy[m, 1], c=col, s=6, alpha=0.5, label=lab)
    ax.legend()
    ax.set_title('Rappresentazioni di stato — colorate per esito')
    fig.tight_layout()
    fig.savefig('states_by_outcome.png', dpi=160)
    print('Salvati states_by_turn.png e states_by_outcome.png')


if __name__ == '__main__':
    main()
