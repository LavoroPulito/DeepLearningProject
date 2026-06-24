import requests # type: ignore
import numpy as np # type: ignore
import torch # type: ignore

class Bitmask:
    def __init__(self,size, valore_iniziale=0):
        """
        Gestisce 19 bit individuali all'interno di un singolo intero Python.
        valore_iniziale può essere un intero (es. da un database) o un'altra maschera.
        """
        self.size = size
        # Maschera di sicurezza per garantire che non si usino più di 19 bit (0x7FFFF = 19 bit a 1)
        self.LIMIT_MASK = (1 << size) - 1  
        self.mask = valore_iniziale & self.LIMIT_MASK

    def set_bit(self, indice, valore):
        """set il bit all'indice specificato (0-18) a 1 (True) o 0 (False)."""
        if not (0 <= indice < self.size):
            raise IndexError("L'indice deve essere compreso tra 0 e 18.")
        
        if valore:
            # Operazione OR per attivare il bit
            self.mask |= (1 << indice)
        else:
            # Operazione AND con il NOT logico per disattivare il bit
            self.mask &= ~(1 << indice)

    def get_bit(self, indice):
        """Ritorna True se il bit all'indice è 1, altrimenti False."""
        if not (0 <= indice < self.size):
            raise IndexError("L'indice deve essere compreso tra 0 e 18.")
        # Sposta il bit a destra e verifica se l'ultimo bit è 1
        return bool((self.mask >> indice) & 1)

    def flip_bit(self, indice):
        """Inverte lo stato del bit (da 0 a 1 e viceversa) usando l'operatore XOR."""
        if not (0 <= indice < self.size):
            raise IndexError("L'indice deve essere compreso tra 0 e 18.")
        self.mask ^= (1 << indice)

    def reset(self):
        """Spegne tutti i bit."""
        self.mask = 0

    def to_list(self):
        """Converte la maschera di bit in una lista classica di 0 e 1 (per compatibilità)."""
        return [self.get_bit(i) for i in range(self.size)]

    def to_tensor(self):
        """Converte la maschera direttamente in un tensore PyTorch (es. per darlo in pasto a una rete)."""
        arr_np = np.array(self.to_list(), dtype=np.float32)
        return torch.from_numpy(arr_np)

    def __str__(self):
        # Rappresentazione binaria formattata a 19 caratteri
        return f"{self.mask:019b}"
    def __repr__(self):
        # Rappresentazione formale dell'oggetto (usata quando l'oggetto è dentro collezioni come liste)
        return self.mask

class Battlefield:
    def __init__(self, turn, winner, current_weather=0, speed_modifier=None, ):
        """
        Rappresenta il campo di battaglia.
        
        :param turn: Turno corrente.
        :param weather: Condizione meteo (default: 0).
        :param speed_modifier: Istanza di Bitmask, intero o None per inizializzare a 0 una Bitmask a 3 bit.
        """
        self.turn = turn
        self.winner = winner
        self.current_weather = current_weather
        
        if speed_modifier is None:
            self.speed_modifier = Bitmask(size=2)  

        elif isinstance(speed_modifier, Bitmask):
            self.speed_modifier = speed_modifier
        else:
            self.speed_modifier = Bitmask(size=2, valore_iniziale=speed_modifier)

    def to_list(self):
        #4 bits
        return [self.turn,self.current_weather,self.speed_modifier.mask,self.winner]
    
replacement = {
               'Aegislash':'Aegislash-shield',
               'Maushold-Four':'maushold-family-of-four',
               'Lycanroc':'Lycanroc-Midday',
               'Sinistcha-Masterpiece':'Sinistcha',
               'Maushold':'maushold-family-of-three',
               'Mimikyu': '778',
               'Morpeko':'877',
               'Palafin': 'palafin-zero'
                       }
to_set_sex = {'Basculegion','Meowstic'}
class Pokemon:
    def fixname(self, name):
        if name.startswith('Vivillon'): 
            name = 'Vivillon'
        elif name in replacement.keys(): 
            name = replacement[name]
        elif name in to_set_sex: 
            name = name + "-female"  # <- Corretto: ora riassegna il valore!
        return name
    
    def __init__(self, player, poke_id, stats_change=None, hp_ratio=1.0, slot=0, seen=0, known_moves=None, item=0, status=None):
        self.player = int(player)
        self.name = poke_id
        pokejs = get_poke_data(pokeUrl + "pokemon/" + self.fixname(poke_id))
        
        self.poke_id = pokejs['id']
        self.stats = get_stat(pokejs['stats'])
        self.types = get_type(pokejs['types'])
        self.ability = get_ability(pokejs['abilities'])
        
        # Gestione sicura delle liste di default per evitare bug di condivisione di memoria
        self.stats_change = stats_change if stats_change is not None else [0, 0, 0, 0, 0]
        self.hp_ratio = hp_ratio 
        self.slot = slot
        self.seen = seen
        self.known_moves = known_moves if known_moves is not None else [0, 0, 0, 0]
        self.item = item
        
        # Gestione sicura della Bitmask
        if status is None:
            self.status = Bitmask(21)
        elif isinstance(status, Bitmask):
            self.status = status
        else:
            self.status = Bitmask(status)

    def to_list(self):
        #4+2+2+6+5+1+4+1 = 25 int
        return [self.player, self.slot, self.seen, self.poke_id] + self.types + [self.ability, self.item] + self.stats + self.stats_change + [self.status.mask] + self.known_moves + [self.hp_ratio]
    
    def add_move(self, move_id):
        for i in range(4):
            if self.known_moves[i] == 0: 
                self.known_moves[i] = move_id
                return 0 
        return -1
    
    # -----------------------------------------------------------------
    # NUOVI METODI DI STAMPA
    # -----------------------------------------------------------------
    def __str__(self):
        # Unisce i tipi (es. "fuego, volador" o semplicemente "agua")
        
        # Filtra le mosse diverse da 0 per mostrare solo quelle effettivamente imparate
        mosse_attive = [m for m in self.known_moves if m != 0]
        mosse_str = ", ".join(map(str, mosse_attive)) if mosse_attive else "Nessuna mossa"
        
        return (
            f"=== Pokémon ===\n"
            f"  ID/Nome      : {self.poke_id} ({self.name})\n"
            f"  Allenatore   : Giocatore {self.player} (Slot: {self.slot}, Visto: {self.seen})\n"
            f"  Tipo         : {self.types} | Abilità: {self.ability}\n"
            f"  PS Residui   : {self.hp_ratio * 100:.1f}%\n"
            f"  Strumento    : {self.item}\n"
            f"  Stato Alter. : {self.status}\n"
            f"  Mosse Conos. : [{mosse_str}]\n"
            f"  Stat Modifier: {self.stats_change}\n"
            f"==============="
        )

    def __repr__(self):
        # Ottimo per le liste: es. Pokemon(P1, ID: 6, Nome: Charizard, HP: 100%)
        return f"Pokemon(Player: {self.player}, ID: {self.poke_id}, Nome: '{self.name}', HP: {self.hp_ratio * 100:.0f}%)"

class Move:
    _api_cache = {}

    def __init__(self, move_name, pl_user, sl_user, pl_target, sl_target):
        if move_name == 0: 
            self.name = ''
            self.sl_user = 0
            self.sl_target = 0
            self.pl_user = 0
            self.pl_target = 0
            self.id = 0
            self.power = 0
            self.priority = 0
            self.type = 0
            self.d_class = 0
            self.t_class = 0
            self.accuracy = 0
        elif move_name == -1:
            self.name = 'switch'
            self.sl_user = sl_user
            self.sl_target = sl_target
            self.pl_user = pl_user
            self.pl_target = pl_target
            self.id = -1
            self.power = 0
            self.priority = 6
            self.type = 0
            self.d_class = 0
            self.t_class = 0
            self.accuracy = 0
        else:
            self.name = move_name
            self.sl_user = sl_user
            self.sl_target = sl_target
            self.pl_user = pl_user
            self.pl_target = pl_target
            
            clean_name = move_name.replace(' ', '-').replace("'", '')
            
            if clean_name in Move._api_cache:
                datajs = Move._api_cache[clean_name] # Lettura istantanea dalla RAM
            else:
                datajs = get_poke_data(pokeUrl + "move/" + clean_name)
                Move._api_cache[clean_name] = datajs # Salviamo il JSON in memoria per la prossima volta
            
            self.d_class = int(datajs['damage_class']['url'].split('/')[-2])
            self.t_class = int(datajs['target']['url'].split('/')[-2])
            self.accuracy = datajs["accuracy"]
            self.id = datajs['id']
            self.power = datajs['power']
            self.priority = datajs['priority']
            self.type = types[datajs['type']['name']]
    
    def to_list(self):
        #11
        return [self.id, self.pl_user,self.sl_user,self.pl_target,self.sl_target,self.type,self.d_class,self.t_class,self.accuracy,self.power,self.priority]

    def __str__(self):
        return f"{self.name}"
    
    def __repr__(self):
        return f"{self.name}"
def get_ability_id(name):
    dati_json = get_poke_data(pokeUrl+'ability/'+name.replace(' ','-'))
    return dati_json['id']

def get_item_id(name):
    dati_json = get_poke_data(pokeUrl+'item/'+name.replace(' ','-'))
    return dati_json['id']




# 1. Inserisci l'URL della pagina o dell'API
pokeUrl = "https://pokeapi.co/api/v2/"
types = {
        "normal"    :1,
        "fighting"  :2,
        "flying"    :3,
        "poison"    :4,
        "ground"    :5,
        "rock"      :6,
        "bug"       :7,
        "ghost"     :8,
        "steel"     :9,
        "fire"      :10,
        "water"     :11,
        "grass"     :12,
        "electric"  :13,
        "psychic"   :14,
        "ice"       :15,
        "dragon"    :16,
        "dark"      :17,
        "fairy"     :18
         }

weather = {
        "SunnyDay"  :1,
        "RainDance" :2,
        "Sandstorm" :3,
        "Snowscape" :4
        }

all_status = {
                  'par': 0,
                  'slp': 1, 
                  'frz': 2,
                  'brn': 3, 
                  'psn': 4, 
            'confusion': 5, 
          'infatuation': 6, 
                 'trap': 7, 
            'nightmare': 8, 
              'torment': 9, 
              'disable': 10, 
                 'yawn': 11, 
           'heal-block': 12, 
     'no-type-immunity': 13, 
           'leech-seed': 14, 
              'embargo': 15, 
          'perish-song': 16, 
              'ingrain': 17,
               'Encore': 18,
                  'tox': 19
                }

stat_code = {
    'atk':0,
    'def':1,
    'spa':2,
    'spd':3,
    'spe':4
}

def get_type(tyls):
    poke_types= [0,0]
    for i,t in enumerate(tyls):
        poke_types[i] = types[t['type']['name']]
    return poke_types

def get_stat(poke_s):
    stats =[]
    for i in range(6):
        stats.append(poke_s[i]['base_stat'])
    return stats

def get_poke_data(targetUrl):
    dati_json = {}
    try:
        # 2. Fai la richiesta GET alla pagina
        risposta = requests.get(targetUrl)
    
        # 3. Controlla che la richiesta sia andata a buon fine (codice 200)
        risposta.raise_for_status() 
    
        # 4. Estrai il JSON e convertilo in un dizionario Python
        dati_json = risposta.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Si è verificato un errore di connessione: {e}")
    except ValueError:
        print("La pagina non ha restituito un JSON valido.")
    return dati_json

def get_ability(abil):
    if len(abil) == 1:
        return int(abil[0]['ability']['url'].split('/')[-2])
    return 0


if __name__ == '__main__':
    a = Battlefield(0)
    print(a.to_list())

'''
inp = "start"
primo = Pokemon(0,'bulbasaur')
secondo = Pokemon(1,'charizard-mega-y')
secondo.add_move(2)
secondo.add_move(4)
secondo.hp_ratio = 0.4
primo.status.set_bit(4,1)
print(primo.to_vector())
print(secondo.to_vector())
print(Bitmask(19,3))
while True:
    inp = input()
    if inp == "e": break
    poke = get_poke_data(pokeUrl+inp)
    print(poke["name"])
    print(poke["stats"][0]['stat']['name'],poke["stats"][0]['base_stat'])
    print(poke["stats"][1]['stat']['name'],poke["stats"][1]['base_stat'])
    print(poke["stats"][2]['stat']['name'],poke["stats"][2]['base_stat'])
    print(poke["stats"][3]['stat']['name'],poke["stats"][3]['base_stat'])
    print(poke["stats"][4]['stat']['name'],poke["stats"][4]['base_stat'])
    print(poke["stats"][5]['stat']['name'],poke["stats"][5]['base_stat'])
    print(get_type(poke['types']))
    print(get_stat(poke['stats']))
    print(poke['abilities'][0]['ability']['url'].split('/')[-2])
'''
