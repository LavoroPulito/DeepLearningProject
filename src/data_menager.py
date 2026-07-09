import numpy as np # type: ignore
import os
import glob
from pathlib import Path
import json

fold1 = Path('../npz/reg_m-A')
fold2 = Path('../npz/reg_m-B')

def _structured_to_token(record):
    """
    It takes a structured numpy record of type _TURN_DT and converts it
    to the original flat list of 544 elements.
    """
    token = []
    

    for i in range(12):
        pk = record['pokemon'][i]
        

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
            

        token.append(pk['hp_ratio'])
        

    field = record['field']
    token.extend([
        field['turn'],
        field['weather'],
        field['speed_mask'],
        field['winner']
    ])
    

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
    Loads an .npz file saved with save_to_npz and returns
    the list of sorted flat tokens (one for each round).
    """
    # Carica i dati dal file compresso
    with np.load(filename) as data:
        turns = data['turns']
        
    # Applica l'inversione a ogni record del turno
    reformatted_tokens = [_structured_to_token(record) for record in turns]
    
    return reformatted_tokens

def collect_data_info():
    id_pokemon = set()
    id_ability = set()
    id_item = set()
    id_moves = set()

    file_npz = list(fold1.glob(f'*.npz'))   
    print(len(file_npz))
    file_npz += list(fold2.glob(f'*.npz'))   
    print(len(file_npz))

    if not file_npz:
        print(f"Warning: No .npz files found in {'../npz/'}")
        exit()
    
    for index, percorso_file in enumerate(file_npz, 1):
        with np.load(percorso_file) as data:
            
            turns = data['turns'] 
            
            pokes = turns['pokemon']
            
            id_pokemon.update(pokes['poke_id'].flatten())
            id_ability.update(pokes['ability'].flatten())
            id_item.update(pokes['item'].flatten())
            
            for slot_mossa in ['move0', 'move1', 'move2', 'move3']:
                id_moves.update(pokes[slot_mossa]['id'].flatten())
                    
        
        if index % 50 == 0 or index == len(file_npz):
            print(f"Elaborati {index}/{len(file_npz)} file...")


    return id_ability, id_moves, id_pokemon, id_item

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
    """Save the all cache in JSON."""
    with open('../data/maps.json', "w", encoding="utf-8") as f:
        json.dump(map, f, indent=4) 

def make_map():
    id_a, id_m, id_p, id_s = collect_data_info() # 'regma' for reg m-A, ow reg m-B
    print_collected_data(id_a,id_m,id_p,id_s)
    l_id_a = list(id_a)
    l_id_a.append(0)
    l_id_a.sort()
    abil_map = {int(l_id_a[i]) : i for i in range(len(l_id_a))}

    l_id_m = sorted(list(id_m))
    move_map = {int(l_id_m[i]) : i for i in range(len(l_id_m))}

    l_id_p = list(id_p)
    l_id_p.append(0)
    l_id_p.sort()
    poke_map = {int(l_id_p[i]) : i for i in range(len(l_id_p))}

    l_id_s = sorted(list(id_s))
    item_map = {int(l_id_s[i]) : i for i in range(len(l_id_s))}

    final_map = {'ability'  : abil_map,
                 'move'     : move_map,
                 'pokemon'  : poke_map,
                 'item'     : item_map}
    
    save_map(final_map)


if __name__ == "__main__":
    make_map()


    '''
    ========================================
          RISULTATI STATISTICHE - reg m-A
    ========================================
    Pokémon unici nel dataset:   241
    Abilità uniche nel dataset:  148
    Strumenti unici nel dataset: 38
    Mosse uniche nel dataset:    322

    ========================================
    VALORI MASSIMI (Per nn.Embedding senza remap)
    ========================================
    ID Max Pokémon:   10321  -> Configura num_embeddings = 10322
    ID Max Abilità:   311  -> Configura num_embeddings = 312
    ID Max Strumenti: 2105  -> Configura num_embeddings = 2106
    ID Max Mosse:     918  -> Configura num_embeddings = 919

    ========================================
          RISULTATI STATISTICHE - reg m-A + reg m-A
    ========================================
    Pokémon unici nel dataset:   296
    Abilità uniche nel dataset:  164
    Strumenti unici nel dataset: 50
    Mosse uniche nel dataset:    389

    ========================================
    VALORI MASSIMI (Per nn.Embedding senza remap)
    ========================================
    ID Max Pokémon:   10321  -> Configura num_embeddings = 10322
    ID Max Abilità:   313  -> Configura num_embeddings = 314
    ID Max Strumenti: 2177  -> Configura num_embeddings = 2178
    ID Max Mosse:     918  -> Configura num_embeddings = 919

    '''