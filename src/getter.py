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
        return f"Bitmask(mask={self.mask})"

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
    

    def fixname(self,name):
        if name.startswith('Vivillon'): name = 'Vivillon'
        elif name in replacement.keys(): name = replacement[name]
        elif name in to_set_sex: name+"female"
        return name
    
    def __init__(self, player,poke_id, stats_change = [0,0,0,0,0], hp_ratio = 1.0, slot = 0, seen = 0, known_moves = [0,0,0,0], item = 0,status = None ):
        self.player = player
        self.name = poke_id
        pokejs = get_poke_data(pokeUrl+"pokemon/"+self.fixname(poke_id))
        
        self.poke_id = pokejs['id']
        self.stats = get_stat(pokejs['stats'])
        self.types = get_type(pokejs['types'])
        self.ability = get_ability(pokejs['abilities'])
        self.stats_change = stats_change
        self.hp_ratio =hp_ratio 
        self.slot = slot
        self.seen = seen
        self.known_moves = known_moves
        self.item = item
        # Gestione sicura del default: se None, creiamo una Bitmask vuota (pari a 0)
        if status is None:
            self.status = Bitmask(19)  # Tutti i 19 bit a 0 di default
        elif isinstance(status, Bitmask):
            self.status = status
        else:
            # Se l'utente passa un numero intero, lo usiamo per inizializzare la maschera
            self.status = Bitmask(status)


    def to_vector(self):
        return [self.player,self.seen, self.slot,self.poke_id] + self.types + [self.ability,self.item]+ self.stats + self.stats_change + [self.status] + self.known_moves + [self.hp_ratio]
    
    def add_move(self,move_id):
        for i in range(4):
            if self.known_moves[i] == 0: 
                self.known_moves[i] = move_id
                return 0 
        return -1
    
    def __str__(self):

        return f"{self.player,self.poke_id}"
    def __repr__(self):

        return f"{self.player,self.poke_id}"



class Battlefield:
    def __init__(self,pokemons, weather = 0, trickroom = False, tailwind = [False, False],lastMove = 0, turn = 0, win = 0, lose = 0):
        self.pokemons = pokemons
        self.weather = weather
        self.trickroom = trickroom
        self.tailwind = tailwind
        self.lastMove = lastMove
        self.turn = turn
        self.win = win
        self.lose = lose
        


class Move:
    def __init__(self,move_name,user, target):
        if move_name == 0: 
            self.user = 0
            self.target = 0
            self.power = 0
            self.id = 0
            self.power = 0
            self.priority = 0
            self.type = 0
        else:
            self.user = user
            self.target = target
            datajs = get_poke_data(pokeUrl+"move/"+move_name.replace(' ','-').replace("'",''))
            self.d_class = int(datajs['damage_class']['url'].split('/')[-2])
            self.t_class = int(datajs['target']['url'].split('/')[-2])
            self.accuracy = datajs["accuracy"]
            self.id = datajs['id']
            self.power = datajs['power']
            self.priority = datajs['priority']
            self.type = types[datajs['type']['name']]
    
    def __str__(self):
        return f"{self.id}"
    
    def __repr__(self):
        return f"{self.id}"

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
