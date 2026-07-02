# Title to be decided
## To do 
- [ ] non tutti gli status sono riconosciuti
- [x] alcune abilità non vengono lette ([from] del avversario)
- [ ] pulire e sistemare il codice che fa schifo
- [x] correzione dei token 
- [ ] organizzazione delle partite per batches
- [ ] controllare e sistemare il codice sull'embedding (mancano tutte le dimensioni)
--- aggiungere nei token?
- [ ] aggiungere mega pietre
- [x] far finire i campi 
- [ ] accuracy non la vediamo
- [ ] aggiungere turno 0

## Domande
- ma le bitmask come funzionano? diventano liste di bit o numeri?

## theory stuff
- [alphago](https://deepmind.google/research/alphago/) 
- [Attention is all you need](https://arxiv.org/pdf/1706.03762) 
- [embedding by medium](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca) 
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

