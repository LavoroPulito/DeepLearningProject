import requests # type: ignore
import numpy as np # type: ignore
import torch # type: ignore

import json
import os
# ==========================================
# CONFIGURATIONS AND CONSTANTS
# ==========================================

pokeUrl = "https://pokeapi.co/api/v2/"

types = {
    "normal": 1, "fighting": 2, "flying": 3, "poison": 4, "ground": 5,
    "rock": 6, "bug": 7, "ghost": 8, "steel": 9, "fire": 10, "water": 11,
    "grass": 12, "electric": 13, "psychic": 14, "ice": 15, "dragon": 16,
    "dark": 17, "fairy": 18
}

weather = {
    "none": 0, "SunnyDay": 1, "RainDance": 2, "Sandstorm": 3, "Snowscape": 4
}

all_status = {
    'par': 0, 'slp': 1, 'frz': 2, 'brn': 3, 'psn': 4, 'tox': 5
}

stat_code = {
    'atk': 0, 'def': 1, 'spa': 2, 'spd': 3, 'spe': 4
}

replacement = {
    'Aegislash': 'Aegislash-shield',
    'Maushold-Four': 'maushold-family-of-four',
    'Lycanroc': 'Lycanroc-Midday',
    'Sinistcha-Masterpiece': 'Sinistcha',
    'Maushold': 'maushold-family-of-three',
    'Mimikyu': '778',
    'Morpeko': '877',
    'Palafin': 'palafin-zero',
    'Basculegion-F': 'Basculegion-female',
    'Tauros-Paldea-Combat':'10250',
    'Tauros-Paldea-Blaze':'10251',
    'Tauros-Paldea-Aqua':'10252',
    'Meowstic-M-Mega':'Meowstic-male-Mega',
    'Basculegion':'Basculegion-male',
    'Basculegion-F':'Basculegion-female',
    'Pyroar':'Pyroar-male',
    "Polteageist-Antique":"855",
    "Polteageist-Antique":"855",
    'Gourgeist': '711'
}

just_begin = {'Vivillon','Florges','Alcremie'} # pokemon that has to be saved just by their "first name"

to_set_sex = {'Basculegion', 'Meowstic'} # pokemon name that need the sex specified

# ==========================================
# GLOBAL CACHE HANDLER AND HELPER API
# ==========================================

CACHE_FILE = "../data/"
os.makedirs(CACHE_FILE, exist_ok=True) 

def load_cache(which):
    file_path = CACHE_FILE + which + '_cache.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f: 
                return json.load(f)
        except json.JSONDecodeError:
            pass 
    return {}

_item_cache = load_cache('item')
_ability_cache = load_cache('ability')

def save_cache(which,cache):
    with open(CACHE_FILE+which+'_cache.json', "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4) 

def get_cache_stats():
    return len(Pokemon._api_cache), len(Move._api_cache),len(_item_cache),len(_ability_cache)
   
def get_poke_data(targetUrl):
    try:
        risposta = requests.get(targetUrl)
        risposta.raise_for_status() 
        return risposta.json()
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
    except ValueError:
        print("No valid JSON.")
    return {}

def get_item_id(name):
    clean_name = name.replace(' ', '-')
    if clean_name in _item_cache:
        return _item_cache[clean_name]
    
    dati_json = get_poke_data(pokeUrl + 'item/' + clean_name)
    item_id = dati_json.get('id', 0)
    _item_cache[clean_name] = item_id
    save_cache('item', _item_cache) 
    return item_id

def get_ability_id(name):
    clean_name = name.replace(' ', '-')
    if clean_name in _ability_cache:
        return _ability_cache[clean_name]
    
    dati_json = get_poke_data(pokeUrl + 'ability/' + clean_name)
    ability_id = dati_json.get('id', 0)
    _ability_cache[clean_name] = ability_id
    save_cache('ability',_ability_cache) 
    return ability_id

def get_type(tyls):
    poke_types = [0, 0]
    for i, t in enumerate(tyls):
        poke_types[i] = types[t['type']['name']]
    return poke_types

def get_stat(poke_s):
    return [poke_s[i]['base_stat'] for i in range(6)]

def get_ability(abil):
    if len(abil) >= 1:
        return int(abil[0]['ability']['url'].split('/')[-2])
    return 0

# ==========================================
# GAME CLASSES
# ==========================================

class Bitmask:
    def __init__(self, size, valore_iniziale=0):
        self.size = size
        self.LIMIT_MASK = (1 << size) - 1  
        self.mask = valore_iniziale & self.LIMIT_MASK

    def set_bit(self, indice, valore):
        if not (0 <= indice < self.size):
            raise IndexError(f"The index must be between 0 and {self.size - 1}.")
        if valore:
            self.mask |= (1 << indice)
        else:
            self.mask &= ~(1 << indice)

    def get_bit(self, indice):
        if not (0 <= indice < self.size):
            raise IndexError(f"The index must be between 0 and {self.size - 1}.")
        return bool((self.mask >> indice) & 1)

    def flip_bit(self, indice):
        if not (0 <= indice < self.size):
            raise IndexError(f"The index must be between 0 and{self.size - 1}.")
        self.mask ^= (1 << indice)

    def reset(self):
        self.mask = 0

    def to_list(self):
        return [self.get_bit(i) for i in range(self.size)]

    def to_tensor(self):
        arr_np = np.array(self.to_list(), dtype=np.float32)
        return torch.from_numpy(arr_np)

    def __str__(self):
        return f"{self.mask:0{self.size}b}"

    def __repr__(self):
        return str(self.mask)


class Battlefield:
    def __init__(self, turn, winner=-1, current_weather=0, speed_modifier=None):
        self.turn = turn
        self.winner = winner
        self.current_weather = current_weather
        
        if speed_modifier is None:
            self.speed_modifier = Bitmask(size=3)  
        elif isinstance(speed_modifier, Bitmask):
            self.speed_modifier = speed_modifier
        else:
            self.speed_modifier = Bitmask(size=3, valore_iniziale=speed_modifier)

    def to_list(self):
        return [self.turn, self.current_weather, self.speed_modifier.mask, self.winner]


class Pokemon:
    _api_cache = load_cache("pokemon") 
    @staticmethod
    def fixname(name):  
        if name in replacement: 
            return replacement[name]
        elif name.split('-')[0] in to_set_sex: 
            return name.split('-')[0] + "-male"  
        else:
            for el in just_begin:
                if name.startswith(el):
                    name = el

        name = name.replace('.','').replace(' ','-')
        return name
    
    def __init__(self, player, poke_id, stats_change=None, hp_ratio=1.0, slot=0, known_moves=None, item=0, status=None):
        self.player = int(player)
        self.name = poke_id
        
        fixed_name = self.fixname(poke_id)
        if fixed_name in Pokemon._api_cache:
            pokejs = Pokemon._api_cache[fixed_name]
        else:
            pokejs = get_poke_data(pokeUrl + "pokemon/" + fixed_name)
            Pokemon._api_cache[fixed_name] = pokejs
            save_cache('pokemon',Pokemon._api_cache)
        
        self.poke_id = pokejs['id']
        self.stats = get_stat(pokejs['stats'])
        self.types = get_type(pokejs['types'])
        self.ability = get_ability(pokejs['abilities'])
        
        self.stats_change = stats_change if stats_change is not None else [0, 0, 0, 0, 0]
        self.hp_ratio = hp_ratio 
        self.slot = slot

        
        self.known_moves = known_moves if known_moves is not None else [Move(0) for _ in range(4)]
        self.item = item
        
        if status is None:
            self.status = Bitmask(6)
        elif isinstance(status, Bitmask):
            self.status = status
        else:
            self.status = Bitmask(6, status)

    def to_list(self):
        moves_list = []
        for m in self.known_moves:
            moves_list += m.to_list()
        return [self.player, self.slot, self.poke_id] + self.types + [self.ability, self.item] + self.stats + self.stats_change + [self.status.mask] + moves_list + [self.hp_ratio]
    
    def add_move(self, new_move):
        '''add a move in the first empty slot of a pokemon moveset (empty mean == 0)'''
        for i in range(4):
            if self.known_moves[i].id == new_move.id:
                return 0
            if self.known_moves[i].id == 0: 
                self.known_moves[i] = new_move
                return 0 
        return -1
    
    def __str__(self):
        mosse_attive = [m for m in self.known_moves if m.id != 0]
        mosse_str = ", ".join(map(str, mosse_attive)) if mosse_attive else "Nessuna mossa"
        
        return (
            f"=== Pokémon ===\n"
            f"  ID/Nome      : {self.poke_id} ({self.name})\n"
            f"  Trainer      : Player {self.player} (Slot: {self.slot})\n"
            f"  Type         : {self.types} | Ability: {self.ability}\n"
            f"  PS           : {self.hp_ratio * 100:.1f}%\n"
            f"  Item         : {self.item}\n"
            f"  Status       : {self.status}\n"
            f"  Known moves  : [{mosse_str}]\n"
            f"  Stat Modifier: {self.stats_change}\n"
            f"==============="
        )

    def __repr__(self):
        return f"Pokemon(Player: {self.player}, ID: {self.poke_id}, Nome: '{self.name}', HP: {self.hp_ratio * 100:.0f}%)"


class Move:
    _api_cache = load_cache("move")

    def __init__(self, move_name):
        if move_name == 0: 
            self.name = ''
            self.id = 0
            self.power = 0
            self.priority = 0
            self.type = 0
            self.d_class = 0
            self.t_class = 0
            self.accuracy = 0
        elif move_name == -1:
            self.name = 'switch'
            self.id = -1
            self.power = 0
            self.priority = 6
            self.type = 0
            self.d_class = 0
            self.t_class = 0
            self.accuracy = 0
        else:
            self.name = move_name
            clean_name = move_name.replace(' ', '-').replace("'", '')
            
            if clean_name in Move._api_cache:
                datajs = Move._api_cache[clean_name] 
            else:
                datajs = get_poke_data(pokeUrl + "move/" + clean_name)
                Move._api_cache[clean_name] = datajs 
                save_cache('move',Move._api_cache)
            
            self.d_class = int(datajs['damage_class']['url'].split('/')[-2])
            self.t_class = int(datajs['target']['url'].split('/')[-2])
            self.accuracy = datajs["accuracy"] if datajs["accuracy"] is not None else 100
            self.id = datajs['id']
            self.power = datajs['power'] if datajs['power'] is not None else 0
            self.priority = datajs['priority']
            self.type = types.get(datajs['type']['name'], 0)
    
    def to_list(self):
        return [self.id, self.type, self.d_class, self.t_class, self.accuracy, self.power, self.priority]

    def __str__(self):
        return f"{self.name}"
    
    def __repr__(self):
        return f"{self.name}"


class Action: 
    def __init__(self, usr_pl, usr_slot, trg_pl, trg_slot, move, mega=0):
        self.usr_slot = int(usr_slot)
        self.trg_slot = int(trg_slot)
        self.usr_pl = int(usr_pl)
        self.trg_pl = int(trg_pl)
        self.move = int(move)
        
        if self.move > 3 and self.move != 6:
            self.mega = 0
        else:
            self.mega = int(mega)

    def to_list(self):
        return [self.usr_pl, self.usr_slot, self.trg_pl, self.trg_slot, self.move, self.mega]

    def __str__(self):
        mega_str = " + Mega" if self.mega == 1 else ""
        return f"Act(from: p{self.usr_pl}{self.usr_slot} to: P{self.trg_pl}{self.trg_slot}; {self.move}){mega_str}"

    def __repr__(self):
        return f"Action({self.usr_pl}, {self.usr_slot}, {self.trg_pl}, {self.trg_slot}, {self.move}, {self.mega})"


if __name__ == '__main__':
    # Test Battlefield, Pokemon, caches
    a = Battlefield(0)
    print("Battlefield List:", a.to_list())
    

    print("Download Charizard (first time)...")
    p1 = Pokemon(0, 'charizard')
    
    print("Download Charizard (first time - from the cache)...")
    p2 = Pokemon(1, 'charizard')
    
    get_item_id("leftovers")
    get_item_id("choice band")
    get_ability_id("intimidate")
    

    get_cache_stats()
