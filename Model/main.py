"""Entry point del training.

Uso:
    python3 Model/main.py                     # training completo
    python3 Model/main.py --fast              # sanity check veloce
    python3 Model/main.py --data npz/reg_m-B --epochs 30 --batch 64

--fast: modello piccolo + poche partite + poche epoche. Serve a verificare
in pochi minuti che la pipeline giri e che il modello impari (la train
accuracy deve salire ben oltre la baseline casuale ~1/50 di azioni legali).
"""
import argparse
import glob
import random
import sys
import time
from pathlib import Path

import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import DecisionTransformer
from TrainingLoop import train_decision_transformer


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(Path(__file__).resolve().parent.parent
                                          / 'npz' / 'reg_m-B'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--max-files', type=int, default=0, help='0 = tutti')
    ap.add_argument('--save-dir', default='checkpoints')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--fast', action='store_true',
                    help='sanity check: modello piccolo, poche partite')
    args = ap.parse_args()

    if args.fast:
        args.epochs = min(args.epochs, 15)
        args.d_model, args.depth, args.heads = 128, 3, 4
        args.max_files = args.max_files or 300

    device = pick_device()
    print(f'Device: {device}')

    files = sorted(glob.glob(f'{args.data}/*.npz'))
    if not files:
        sys.exit(f'Nessun file .npz in {args.data}')
    random.Random(42).shuffle(files)
    if args.max_files:
        files = files[:args.max_files]

    n = len(files)
    n_val, n_test = max(1, n // 10), max(1, n // 10)
    train_files = files[:n - n_val - n_test]
    val_files = files[n - n_val - n_test:n - n_test]
    test_files = files[n - n_test:]
    print(f'Partite: {len(train_files)} train / {len(val_files)} val / '
          f'{len(test_files)} test')

    t0 = time.time()
    ds_train = PokemonVGCDataset(train_files, max_turn=49, verbose=True,
                                 augment=not args.no_augment)
    ds_val = PokemonVGCDataset(val_files, max_turn=49)
    print(f'Preprocessing: {time.time() - t0:.1f}s (una tantum, poi in RAM)')

    pin = device.type == 'cuda'
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=pin, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=pin)

    model = DecisionTransformer(action_dim=360, d_model=args.d_model,
                                n_heads=args.heads, depth=args.depth,
                                max_turn=49, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parametri: {n_params / 1e6:.2f}M')

    train_decision_transformer(model, dl_train, dl_val,
                               num_epochs=args.epochs, device=device,
                               lr=args.lr, save_dir=args.save_dir,
                               resume_from=args.resume)

    torch.save(model.state_dict(), 'vgc_decision_transformer.pth')
    print('Modello salvato: vgc_decision_transformer.pth')


if __name__ == '__main__':
    main()
