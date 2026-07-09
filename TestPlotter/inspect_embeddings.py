"""Ispezione degli embedding imparati (pokemon e mosse).

Estrae le matrici embed_id / embed_id_move dai pesi addestrati, rimappa ogni
riga al nome reale (via data/pokemon_cache.json e move_cache.json) e permette:

  vicini più prossimi (cosine) di un pokemon/mossa:
      python3 Model/inspect_embeddings.py pesi.pth --what pokemon --query Incineroar
      python3 Model/inspect_embeddings.py pesi.pth --what move --query Protect

  i 10 vicini di TUTTE le voci piu' "isolate"/"centrali" e mappa 2D (PCA):
      python3 Model/inspect_embeddings.py pesi.pth --what pokemon --plot
      python3 Model/inspect_embeddings.py pesi.pth --what move --plot

Nota interpretativa: tipo, stats, ability e item entrano nel token come
feature SEPARATE, quindi embed_id impara solo l'informazione residua
dell'identita' (ruolo, stile d'uso, sinergie) non gia' spiegata da quelle
feature. Cluster per tipo possono comunque emergere, ma similarita' "di
ruolo" (es. due support diversi vicini) sono il segnale piu' interessante.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from id_maps import pokemon_map, move_map

ROOT = Path(__file__).resolve().parent.parent


def load_names(what):
    """raw_id -> (nome, categoria per il colore del plot)."""
    if what == 'pokemon':
        cache = json.load(open(ROOT / 'data' / 'pokemon_cache.json'))
        out = {}
        for k, v in cache.items():
            if isinstance(v, dict) and 'id' in v:
                t = v['types'][0]['type']['name'] if v.get('types') else '?'
                out[int(v['id'])] = (k, t)
        return out
    cache = json.load(open(ROOT / 'data' / 'move_cache.json'))
    out = {}
    for k, v in cache.items():
        if isinstance(v, dict) and 'id' in v:
            t = (v.get('type') or {}).get('name', '?')
            dc = (v.get('damage_class') or {}).get('name', '?')
            out[int(v['id'])] = (f'{k}', f'{t}/{dc}')
    return out


def load_matrix(weights_path, what):
    sd = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    key = ('token_embedding.embed_id.weight' if what == 'pokemon'
           else 'token_embedding.embed_id_move.weight')
    W = sd[key].float().numpy()

    id_map = pokemon_map if what == 'pokemon' else move_map
    names = load_names(what)
    rows, labels, cats = [], [], []
    for raw, emb in id_map.items():
        name, cat = names.get(int(raw), (f'id{raw}', '?'))
        rows.append(emb + 1)          # +1: indice 0 = sconosciuto/padding
        labels.append(name)
        cats.append(cat)
    return W[rows], labels, cats


def cosine_matrix(W):
    Wc = W - W.mean(axis=0)           # centrato: toglie la direzione comune
    n = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-8)
    return n @ n.T


def neighbors(sim, labels, query, k=12):
    idx = [i for i, l in enumerate(labels) if l.lower() == query.lower()]
    if not idx:
        cand = [l for l in labels if query.lower() in l.lower()]
        sys.exit(f"'{query}' non trovato. Forse: {cand[:8]}")
    i = idx[0]
    order = np.argsort(-sim[i])
    print(f'\nVicini di {labels[i]} (cosine, embedding centrato):')
    for j in order[1:k + 1]:
        print(f'  {sim[i, j]:+.3f}  {labels[j]}')
    print(f'\nPiu\' lontani:')
    for j in order[-5:]:
        print(f'  {sim[i, j]:+.3f}  {labels[j]}')


def plot(W, labels, cats, what, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Wc = W - W.mean(axis=0)
    # PCA via SVD
    U, S, _ = np.linalg.svd(Wc, full_matrices=False)
    xy = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2].sum()

    uniq = sorted(set(cats))
    cmap = plt.get_cmap('tab20', len(uniq))
    color = {c: cmap(i) for i, c in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(16, 12))
    for c in uniq:
        pts = np.array([p for p, cc in zip(xy, cats) if cc == c])
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=color[c], label=c)
    for (x, y), l in zip(xy, labels):
        ax.annotate(l, (x, y), fontsize=5, alpha=0.75)
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.set_title(f'Embedding {what} — PCA 2D ({var:.0%} varianza spiegata)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f'Salvato {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('weights', help='path a .pth (state_dict o checkpoint)')
    ap.add_argument('--what', choices=['pokemon', 'move'], default='pokemon')
    ap.add_argument('--query', help='nome di cui mostrare i vicini')
    ap.add_argument('--plot', action='store_true', help='salva la mappa PCA 2D')
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    W, labels, cats = load_matrix(args.weights, args.what)
    print(f'{len(labels)} {args.what} | embedding dim {W.shape[1]} | '
          f'norma media {np.linalg.norm(W, axis=1).mean():.3f}')

    sim = cosine_matrix(W)
    if args.query:
        neighbors(sim, labels, args.query, args.top)
    else:
        # le 5 coppie piu' simili in assoluto: colpo d'occhio sulla struttura
        iu = np.triu_indices(len(labels), 1)
        top = np.argsort(-sim[iu])[:15]
        print('\nCoppie piu\' simili:')
        for t in top:
            i, j = iu[0][t], iu[1][t]
            print(f'  {sim[i, j]:+.3f}  {labels[i]}  ~  {labels[j]}')

    if args.plot:
        plot(W, labels, cats, args.what,
             f'embeddings_{args.what}.png')


if __name__ == '__main__':
    main()
