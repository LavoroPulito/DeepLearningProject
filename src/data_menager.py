import numpy as np # type: ignore
import os
import glob
from pathlib import Path
import json

cartella = Path('../npz/')


def _structured_to_token(record):

    """
    Prende un record numpy strutturato di tipo _TURN_DT e lo converte
    nella lista piatta originale di 544 elementi.
    """
    token = []
    
    # 1. Ricostruisci i 12 Pokémon
    for i in range(12):
        pk = record['pokemon'][i]
        
        # Primi 19 campi base del Pokémon
        token.extend([
            pk['player'],
            pk['slot'],
            pk['poke_id'],
            pk['type1'],
            pk['type2'],
            pk['ability'],
            pk['item'],
            pk['hp_base'],
            pk['atk'],
            pk['def_'],
            pk['spa'],
            pk['spd'],
            pk['spe'],
            pk['atk_c'],
            pk['def_c'],
            pk['spa_c'],
            pk['spd_c'],
            pk['spe_c'],
            pk['status_mask']
        ])
        
        # Le 4 mosse (7 campi ciascuna)
        for mname in ('move0', 'move1', 'move2', 'move3'):
            m = pk[mname]
            token.extend([
                m['id'],
                m['type'],
                m['d_class'],
                m['t_class'],
                m['accuracy'],
                m['power'],
                m['priority']
            ])
            
        # Ultimo campo: hp_ratio
        token.append(pk['hp_ratio'])
        
    # 2. Ricostruisci il Field (4 campi)
    field = record['field']
    token.extend([
        field['turn'],
        field['weather'],
        field['speed_mask'],
        field['winner']
    ])
    
    # 3. Ricostruisci le 2 Azioni (6 campi ciascuna)
    for aname in ('action0', 'action1'):
        a = record[aname]
        token.extend([
            a['usr_pl'],
            a['usr_slot'],
            a['trg_pl'],
            a['trg_slot'],
            a['move'],
            a['mega']
        ])
        
    return token

def load_from_npz(filename):
    """
    Carica un file .npz salvato con save_to_npz e restituisce
    la lista di token piatti ordinati (uno per ogni turno).
    """
    # Carica i dati dal file compresso
    with np.load(filename) as data:
        turns = data['turns']
        
    # Applica l'inversione a ogni record del turno
    tokens_ricostruiti = [_structured_to_token(record) for record in turns]
    
    return tokens_ricostruiti

def collect_data_info(format):
    id_pokemon = set()
    id_abilita = set()
    id_strumenti = set()
    id_mosse = set()



    # Trova tutti i file .npz nella cartella
    file_npz = list(cartella.glob(f'*{format}*.npz'))   

    if not file_npz:
        print(f"Attenzione: Nessun file .npz trovato in {'../npz/'}")
        exit()
    
    for indice, percorso_file in enumerate(file_npz, 1):
        with np.load(percorso_file) as data:
            # Carichiamo l'array strutturato dei turni
            turns = data['turns'] 
            
            # Accediamo alla sezione pokemon: ha forma (Numero_Turni, 12)
            pokes = turns['pokemon']
            
            # .flatten() trasforma la matrice in un unico vettore piatto per estrarre tutto insieme
            id_pokemon.update(pokes['poke_id'].flatten())
            id_abilita.update(pokes['ability'].flatten())
            id_strumenti.update(pokes['item'].flatten())
            
            # Estraiamo gli ID delle mosse da tutti e 4 gli slot
            for slot_mossa in ['move0', 'move1', 'move2', 'move3']:
                id_mosse.update(pokes[slot_mossa]['id'].flatten())
                    
        # Stampa un aggiornamento ogni 50 file elaborati
        # if indice % 50 == 0 or indice == len(file_npz):
        #     print(f"Elaborati {indice}/{len(file_npz)} file...")


    return id_abilita, id_mosse, id_pokemon, id_strumenti

def print_collected_data(id_a, id_m, id_p, id_s):
    print("\n" + "="*40)
    print("          RISULTATI STATISTICHE")
    print("="*40)

    # 3. Mostra il numero di elementi UNICI (quanti ce ne sono nel dataset)
    print(f"Pokémon unici nel dataset:  {len(id_p)}")
    print(f"Abilità uniche nel dataset: {len(id_a)}")
    print(f"Strumenti unici nel dataset:{len(id_s)}")
    print(f"Mosse uniche nel dataset:   {len(id_m)}")

    print("\n" + "="*40)
    print("   VALORI MASSIMI (Per nn.Embedding senza remap)")
    print("="*40)

    # 4. Mostra il valore MASSIMO trovato + il valore da inserire in num_embeddings
    # Nota: se usi 0 o -1 come flag per "Nessuno strumento/mossa", max() lo gestirà correttamente
    max_poke = max(id_p) if id_p else 0
    max_abi  = max(id_a) if id_a else 0
    max_str  = max(id_s) if id_s else 0
    max_mos  = max(id_m) if id_m else 0

    print(f"ID Max Pokémon:   {max_poke}  -> Configura num_embeddings = {max_poke + 1}")
    print(f"ID Max Abilità:   {max_abi}  -> Configura num_embeddings = {max_abi + 1}")
    print(f"ID Max Strumenti: {max_str}  -> Configura num_embeddings = {max_str + 1}")
    print(f"ID Max Mosse:     {max_mos}  -> Configura num_embeddings = {max_mos + 1}")

def save_map(map):
    """Salva l'intero stato della cache nel file JSON."""
    with open('../data/maps.json', "w", encoding="utf-8") as f:
        json.dump(map, f, indent=4) 

if __name__ == "__main__":
    id_a, id_m, id_p, id_s = collect_data_info('regma') # 'regma' for reg m-A, ow reg m-B

    l_id_a = list(id_a)
    l_id_a.append(0)
    l_id_a.sort()
    abil_map = {int(l_id_a[i]) : i for i in range(len(l_id_a))}

    l_id_m = sorted(list(id_m))
    move_map = {int(l_id_m[i]) : i for i in range(len(l_id_m))}

    l_id_p = sorted(list(id_p))
    poke_map = {int(l_id_p[i]) : i for i in range(len(l_id_p))}

    l_id_s = sorted(list(id_s))
    item_map = {int(l_id_s[i]) : i for i in range(len(l_id_s))}

    final_map = {'ability'  : abil_map,
                 'move'     : move_map,
                 'pokemon'  : poke_map,
                 'item'     : item_map}
    
    save_map(final_map)

    '''
    ========================================
          RISULTATI STATISTICHE - reg m-A
    ========================================
    Pokémon unici nel dataset:  241
    Abilità uniche nel dataset: 148
    Strumenti unici nel dataset:38
    Mosse uniche nel dataset:   322

    ========================================
    VALORI MASSIMI (Per nn.Embedding senza remap)
    ========================================
    ID Max Pokémon:   10321  -> Configura num_embeddings = 10322
    ID Max Abilità:   311  -> Configura num_embeddings = 312
    ID Max Strumenti: 2105  -> Configura num_embeddings = 2106
    ID Max Mosse:     918  -> Configura num_embeddings = 919




    '''