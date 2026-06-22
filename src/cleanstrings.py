from getter import * 
import os
import re
from collections import defaultdict


def moves_setupper(allRows):
    for r in [r for r in allRows if r[0] == 'move']:
        
        name = r[2].replace(' ','-').replace("'",'')

        if name in fatti: break
        print(name)
        fatti.add(name)
        Move(name,0,0)
    return len(fatti)

def setupper(allRows):
    pokemons = []
    positions = {}
    for r in [r for r in allRows if r[0] == 'poke']:
        
        name = r[2].split(',')[0]
        
        if name.startswith('Vivillon'):name = 'Vivillon'
        if name in fatti: break
        print(name)
        fatti.add(name)
        if name in replacement.keys(): name = replacement[name]
        if name in to_set_sex:
            if r[2].split(',')[2] == 'F':
                name+='-female'
            else: 
                name+='-male'
        pokemons.append(Pokemon(int(r[1][1])-1,name))
    return pokemons


def filter_line(list_of_lines):
    p1_name = list_of_lines[0][4:]
    p2_name = list_of_lines[1][4:]

    interesting_tags = {"poke", "-ability",'turn', 'move', '-damage', 'detailschange','-sidestart','-sideend','-enditem','-weather', 'win', '-heal', '-boost','-unboost','-status','-start'}

    row_of_interest = []
    for r in list_of_lines:
        r = r.replace(p1_name,"0")
        r = r.replace(p2_name,"1")
        rl = r.split('|')
        if len(rl)>=2:
            if rl[1] in interesting_tags:
                row_of_interest.append(rl[1:])
    return row_of_interest

def convert_log(list_of_lines):
    list_of_lines = filter_line(list_of_lines)
    turn = 0
    #set pokemon
    mons = []
    for line in [p for p in list_of_lines if p[0] == 'poke']:
        pkName = line[2].split(',')[0]
        
        pkmn = Pokemon(int(line[1][1])-1,pkName)
        mons.append(pkmn)
        for pkmn in mons:
            list_of_lines = [
            [el.replace(pkmn.name, str(pkmn.poke_id)) if isinstance(el, str) else el
             for el in line]
            for line in list_of_lines
        ]

    info = parse_battle_log(list_of_lines)
    for el in info: 
        print(info)

    # for line in list_of_lines:


    #     if line[0] == 'turn':
    #         turn = line[1]

    #     elif line[0] == 'switch':
    #         pass           
     
def parse_battle_log(log_lines):
    pokemon_info = defaultdict(lambda: {
        'ability': None,
        'item': None,
        'moves': set()
    })

    for line in log_lines:
        if not line:
            continue

        tag = line[0]

        # abilità esplicita: ['-ability', 'p1a: Floette', 'Fairy Aura']
        if tag == '-ability':
            pokemon = normalize_name(line[1])
            ability = line[2]
            pokemon_info[pokemon[0]+pokemon[1]]['ability'] = ability

        # abilità da [from] ability: Sand Stream
        # appare in qualsiasi tag come ultimo elemento tipo '[from] ability: Sand Stream'
        for field in line[1:]:
            if isinstance(field, str) and '[from] ability:' in field:
                ability_name = field.split('[from] ability:')[1].strip()
                # il pokemon che ha l'abilità è in '[of] pXx: Nome'
                of_field = next((f for f in line if isinstance(f, str) and '[of]' in f), None)
                if of_field:
                    pokemon = normalize_name(of_field.replace('[of] ', ''))
                    pokemon_info[pokemon[0]+pokemon[1]]['ability'] = ability_name

        # mosse
        if tag == 'move':
            pokemon = normalize_name(line[1])
            move = line[2]
            pokemon_info[pokemon[0]+pokemon[1]]['moves'].add(move)

        # item consumato: ['-enditem', 'p2a: Sneasler', 'White Herb']
        elif tag == '-enditem':
            pokemon = normalize_name(line[1])
            item = line[2]
            pokemon_info[pokemon[0]+pokemon[1]]['item'] = item

        # item da [from] item: appare come campo in vari tag
        # es. ['-heal', 'p1a: Floette', '64/100', '[from] item: Leftovers']
        for field in line[1:]:
            if isinstance(field, str) and '[from] item:' in field:
                item_name = field.split('[from] item:')[1].strip()
                # il pokemon è sempre il secondo elemento della riga
                if len(line) > 1 and isinstance(line[1], str) and re.match(r'p\d[ab]:', line[1]):
                    pokemon = normalize_name(line[1])
                    pokemon_info[pokemon[0]+pokemon[1]]['item'] = item_name

    return pokemon_info

def normalize_name(raw):
    # "p1a: Sneasler" → "p1: Sneasler"
    match = re.match(r'(p\d)[ab]: (.+)', raw)
    if match:
        player = match.group(1)
        name = match.group(2)
        return [player, name]
        #return f"{player}: {name}"
    return raw

def print_report(pokemon_info):
    # raggruppa per player
    players = defaultdict(dict)
    for key, data in pokemon_info.items():
        player = key.split(':')[0].strip()
        name = key.split(':')[1].strip()
        players[player][name] = data

    for player in sorted(players.keys()):
        print(f"\n{'='*40}")
        print(f"  {player.upper()}")
        print(f"{'='*40}")
        for name, data in sorted(players[player].items()):
            print(f"\n  {name}")
            print(f"    Abilità : {data['ability'] or 'non rilevata'}")
            print(f"    Strumento: {data['item'] or 'non rilevato'}")
            moves = sorted(data['moves'])
            if moves:
                print(f"    Mosse   : {', '.join(moves)}")
            else:
                print(f"    Mosse   : nessuna registrata")


if __name__ == "__main__":  

    existent_logs = os.listdir("../logs/")
    for logfile in existent_logs[:1]:
        file = open("../logs/"+logfile)
        convert_log(file.read().split('\n'))
    





