# Metamon: Behavioral Cloning for the Pokémon VGC Meta-game

An offline reinforcement learning agent for competitive Pokémon VGC double battles
(Gen 9, Regulations M-A / M-B, Mega Evolutions), trained on human replays from
Pokémon Showdown. The model is a **Decision Transformer**
([Chen et al., 2021](https://arxiv.org/pdf/2106.01345)): battles are modelled as
sequences of `(return-to-go, state, action)` tokens and the agent learns to predict
the actions played by human players, conditioned on the desired game outcome.

Here you can find the [report]{/Report/report.pdf} to the project

---

## Project Overview

The pipeline consists of three stages:

1. **Data collection** — `Scraper/scraper.py` downloads public VGC replays from Pokémon
   Showdown; `Scraper/cleanstrings.py` and `Scraper/getter.py` parse the battle logs
   (enriched with PokéAPI metadata, cached in `data/`) into structured NumPy arrays,
   one `.npz` file per game (stored in `npz/<regulation>/`).
2. **Preprocessing** — `Model/preprocess.py` converts each game into padded tensors,
   remaps all IDs to compact embedding indices, computes the per-turn legal action
   mask and the flat action targets. Fully vectorised (~2 ms per game); results are
   cached in RAM by the Dataset.
3. **Training** — `Model/TrainingLoop.py` / `notebook/deeplearning.ipynb` train
   the Decision Transformer with mixed precision, optional multi-GPU support,
   data augmentation and a cosine learning-rate schedule.

## Repository Structure

````
├── Scraper/                 # Scraping and replay parsing
│   ├── scraper.py           # Showdown replay downloader
│   ├── cleanstrings.py      # Log parser → structured turns
│   ├── getter.py            # PokéAPI entities (Pokémon, moves, bitmasks)
│   └── data_menager.py      # data reader and maps maker
├── data/                    # PokéAPI caches (pokemon, moves, abilities, items)
├── npz/                     # One .npz per game, grouped by regulation
├── Model/
│   ├── id_maps.py           # Raw ID → embedding index lookup tables
│   ├── preprocess.py        # Vectorised game preprocessing + data augmentation
│   ├── LegalActionMask.py   # Per-turn legal action mask (action space 360)
│   ├── PokemonVGCDataset.py # PyTorch Dataset with in-RAM cache
│   ├── Embedding.py         # State / action / return / turn token embeddings
│   ├── SelfAttention.py     # Transformer blocks (Flash Attention, padding-safe)
│   ├── DecisionTransformer.py
│   ├── TrainingLoop.py      # AMP, legal-action masking, checkpointing, LR schedule
│   ├── main.py              # Local CLI entry point
└── notebook/
    └── deeplearning.ipynb    # Kaggle notebook (2× GPU)
````

## Model

Decision Transformer with GPT-style causal attention over interleaved
`(R̂_t, s_t, a_t)` tokens (sequence length 3 × 49):

- Per-feature embeddings (dim 16) concatenated and projected to `d_model`;
  discrete IDs are remapped by the Dataset (index 0 = unknown/padding).
- `depth` pre-norm Transformer blocks with Flash Attention
  (`scaled_dot_product_attention`) and a padding-safe attention mask.
- A linear head reads the state token and outputs two independent
  distributions over the 360 actions (one per action slot).

Default configuration: `d_model = 384`, `depth = 8`, `n_heads = 12`,
`dropout = 0.20` (~17,46 M parameters).

## Training

- **Objective:** masked negative log-likelihood over the legal actions of both
  action heads, ignoring padded turns.
- **Optimisation:** AdamW, cosine LR decay (optional linear warm-up),
  gradient clipping, automatic mixed precision on CUDA.
- **Multi-GPU:** `nn.DataParallel` (autocast applied inside `forward`, checkpoints
  always stored unwrapped for single/multi-GPU compatibility).
- **Data augmentation:** random permutation of the 12 Pokémon rows (position in
  the token is arbitrary) and of each moveset order, with consistent remapping of
  action targets and mask columns.
- **Fine-tuning mode** (notebook, `FINETUNE = True`): loads pretrained weights,
  reports the zero-shot baseline, then trains with low LR, warm-up, and the
  backbone frozen for the first epochs (last block + head only). Intended for
  regulation transfer (e.g. pretrain on M-A, adapt to M-B).

### Running

Kaggle (recommended, 2× T4): upload the `.npz` dataset, open
`notebook/deeplearning_v2.ipynb`, set the flags in the configuration cell
(`FAST`, `FINETUNE`, paths) and run all cells. Checkpoints are written to
`/kaggle/working/checkpoints_*` and training resumes automatically.

Local:

````bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 Model/main.py --fast           # quick run: small model, 300 games
python3 Model/main.py                  # full training
````


## Known Limitations

- Replays expose only public information: opponent movesets, items and abilities
  are revealed progressively; the state snapshot is taken at the start of the
  turn and cannot capture intra-turn dynamics. The legal mask is therefore
  deliberately permissive.
- The recorded targets of self/spread moves are noisy (~5–7 % of actions).
- The return signal is binary and constant per game; empirically the model
  largely ignores the return token and behaves as behavioural cloning. A richer
  return (e.g. final Pokémon differential) is the natural extension.
- Evaluation is offline only; assessing actual playing strength would require
  rollouts against a battle simulator (e.g. Pokémon Showdown).

## References

- Chen et al., *Decision Transformer: Reinforcement Learning via Sequence
  Modeling*, NeurIPS 2021 — https://arxiv.org/pdf/2106.01345
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Pokémon Showdown (replays) — https://pokemonshowdown.com
- PokéAPI (game metadata) — https://pokeapi.co
