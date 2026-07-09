import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore

try:
    from .preprocess import preprocess_game, augment_game
except ImportError:
    from preprocess import preprocess_game, augment_game


class PokemonVGCDataset(Dataset):
    """VGC game dataset.

            Each game is preprocessed once (vectorized numpy,
            ~2 ms/game) and held in RAM: subsequent epochs only cost
            indexing + torch.from_numpy (zero-copy). With the cache in RAM,
            use num_workers=0 in the DataLoader: this is already the fastest path
            and avoids duplicating the cache in the workers.
    """

    def __init__(self, file_paths, max_turn=49, preload=True, verbose=False,
                 augment=False, seed=None):
        self.file_paths = list(file_paths)
        self.max_turn = max_turn
        self.augment = augment   
        self._rng = np.random.default_rng(seed)
        self._cache = [None] * len(self.file_paths)
        if preload:
            for i in range(len(self.file_paths)):
                self._cache[i] = preprocess_game(self.file_paths[i], max_turn)
                if verbose and (i + 1) % 200 == 0:
                    print(f'  preprocess {i + 1}/{len(self.file_paths)}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        g = self._cache[idx]
        if g is None:
            g = self._cache[idx] = preprocess_game(self.file_paths[idx],
                                                   self.max_turn)
        if self.augment:
            g = augment_game(g, self._rng)
        t = torch.from_numpy
        return {
            'state': {k: t(v) for k, v in g['state'].items()},
            'move': {k: t(v) for k, v in g['move'].items()},
            'battlefield': {k: t(v) for k, v in g['battlefield'].items()},
            'action': {k: t(v) for k, v in g['action'].items()},
            'reward': t(g['reward']),
            'turn': t(g['turn']),
            'padding_mask': t(g['padding_mask']),
            'target_flat': t(g['target_flat']),          # (max_turn, 2)
            'legal_action_mask': t(g['legal_action_mask']),  # (max_turn, 360) bool
        }


if __name__ == '__main__':
    import glob
    files = sorted(glob.glob('../npz/reg_m-B/*.npz'))[:4]
    ds = PokemonVGCDataset(files)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    b = next(iter(dl))
    print('state.id', b['state']['id'].shape)          # (2, 49, 12)
    print('target_flat', b['target_flat'].shape)       # (2, 49, 2)
    print('legal', b['legal_action_mask'].shape, b['legal_action_mask'].dtype)
    tf = b['target_flat']
    legal = b['legal_action_mask']
    ok = legal.gather(-1, tf).all()
    print('target sempre legale:', bool(ok))
