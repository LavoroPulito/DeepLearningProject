# Title to be decided
## To do 
- [x] non tutti gli status sono riconosciuti
- [x] alcune abilità non vengono lette ([from] del avversario)
- [x] pulire e sistemare il codice che fa schifo
- [x] correzione dei token 
- [ ] organizzazione delle partite per batches
- [x] controllare e sistemare il codice sull'embedding (mancano tutte le dimensioni)
- [x] spacchettare le bitmask
- [x] mappare gli id 

--- aggiungere nei token?
- [ ] aggiungere mega pietre
- [x] far finire i campi 
- [x] accuracy non la vediamo
- [x] aggiungere turno 0
- [x] maschera illegalità


## theory stuff
- [alphago](https://deepmind.google/research/alphago/) 
- [Attention is all you need](https://arxiv.org/pdf/1706.03762) 
- [embedding by medium](https://medium.com/deeper-learning/,glossary-of-deep-learning-word-embedding-f90c3cec34ca) 
- [embedding 2 by medium](https://medium.com/data-science/sequence-embedding-for-clustering-and-classification-f816a66373fb)
- [Transformer](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Decision Transformers](https://arxiv.org/pdf/2106.01345) 

Youtube:
- [ Attention ](https://www.youtube.com/watch?v=RNF0FvRjGZk&t=2s)
- [ Transformer](https://www.youtube.com/watch?v=wjZofJX0v4M)

## Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### instruction to first use 
git clone <repo>
cd <repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scraper.py


## Costruzione del modello

cosa trovi nel token


token
 - stato:
    - x12 pokemon:
        - id 
        - type
        - ability
        - item
        - slot 
      
        --continue
        - stats
        - stats_change
        - status
        - hp_ratio
        - 4x move
            - id 
            - d_class
            - t_class

            --continue
            - power
            - priority
            - accuracy
    - campo
        - meteo

        --continue
        - speed_modifier [taw 0, taw 1,tkrm] 

 - turno:
    - (timestamp)

 - 2x azione: 
    - player user
    - slot user
    - player target
    - slot target
    - move $\in  \{0,1,2,3,4,5\}$
    - mega

 - reward
    {0,1}


matrice emb: batch, turns, token = (pokemon, campo, turn, action, reward) 

### legal actions

Spazio azioni piatto = 360: dims (s_user: 3, p_target: 2, s_target: 5, mega: 2, move: 6),
flat = s_user*120 + p_target*60 + s_target*12 + mega*6 + move.
p_user non è una dimensione (sempre 0 nei dati). Convenzioni dei replay:
pass = (s_user=0, trg=(0,0), move=5); switch (move=4) ha trg_slot = slot che
il mon entrante aveva prima (0 = mai sceso in campo, 3/4 = panchina).

Regole (vedi Model/LegalActionMask.py, validate su tutti i replay con
Model/validate_mask.py — copertura 99.55%, il resto è forzato legale dal Dataset):
- pass sempre disponibile
- switch: verso 3/4 se panchina viva (o possibile benching intra-turno), verso 0 sempre (bring-4)
- mosse: serve un mon proprio nello slot; id==0 legale solo al primo indice
  libero (mossa rivelata al primo uso); tutti i 4 target (i target dello
  scraper per mosse self/spread sono rumorosi)
- mega: solo se nessuna azione precedente della partita ha già megaevoluto

La maschera è volutamente permissiva: un -inf sull'azione vera renderebbe la
loss infinita. Il Dataset forza comunque mask[azione_vera] = True.
