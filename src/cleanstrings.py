from getter import *
import os
import re
from collections import defaultdict
import copy
import csv

def save_to_csv(tokens, filename, fill_none=0):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tokens:
            writer.writerow([fill_none if v is None else v for v in row])


# ─── costanti ────────────────────────────────────────────────
INTERESTING_TAGS = {
    "poke", "-ability", "turn", "move", "-damage", "detailschange","switch",
    "-sidestart", "-sideend", "-enditem", "-weather", "win",
    "-heal", "-boost", "-unboost", "-status", "-start"
}

# ─── parsing ─────────────────────────────────────────────────

def parse_field(field):
    """Splitta un campo su ':' o ',' restituendo una lista di token puliti."""
    tokens = re.split(r'[:,]', field)
    return [t.strip() for t in tokens if t.strip()]

def filter_lines(raw_lines):
    if len(raw_lines) < 2:
        return []

    p1_name = raw_lines[0][4:]
    p2_name = raw_lines[1][4:]

    result = []
    for raw in raw_lines:
        raw = raw.replace(p1_name, "0").replace(p2_name, "1")
        # NON sostituire p1a/p2b qui — lascia il formato originale
        # così extract_pokemon_id continua a funzionare
        parts = raw.split('|')
        if len(parts) >= 2 and parts[1] in INTERESTING_TAGS:
            result.append(parts[1:])

    return result

def extract_pokemon_id(field):
    match = re.match(r'p(\d)([ab]):\s*(.+)', field)
    if match:
        player = int(match.group(1)) - 1  # p1→0, p2→1
        position = match.group(2)
        name = match.group(3).strip()
        return player, position, name
    return None

# ─── sostituzione nomi → id ───────────────────────────────────

def replace_in_line(raw, substitutions):
    sorted_subs = sorted(substitutions.items(), key=lambda x: len(x[0]), reverse=True)
    for old, new in sorted_subs:
        # usa word boundary per evitare sostituzioni parziali
        # ma i nomi pokemon hanno trattini quindi usiamo lookahead/lookbehind su caratteri non-alfanumerici
        raw = re.sub(r'(?<![A-Za-z0-9\-])' + re.escape(old) + r'(?![A-Za-z0-9\-])', str(new), raw)
    return raw

def build_nickname_map(raw_lines):
    """
    Costruisce un dizionario {soprannome: nome_reale} leggendo
    le righe 'poke' (per il nome) e 'switch' (per il soprannome).
    """
    nickname_map = {}
    
    # mappa posizione → nome reale dalle righe 'poke'
    # 'poke' ha il nome reale in line[2] (es. "Delphox, L50, F")
    position_to_name = {}
    for raw in raw_lines:
        parts = raw.split('|')
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == 'poke':
            # |poke|p1|Delphox, L50, F|
            player_slot = parts[2]          # 'p1'
            real_name = parts[3].split(',')[0].strip()
            position_to_name[player_slot] = real_name

        elif tag == 'switch':
            # |switch|p1b: 比忍蛙强|Delphox, L50, F|100/100
            ref = parts[2]                  # 'p1b: 比忍蛙强'
            real_name = parts[3].split(',')[0].strip()
            
            match = re.match(r'p(\d)[ab]:\s*(.+)', ref)
            if match:
                nickname = match.group(2).strip()
                if nickname != real_name:
                    nickname_map[nickname] = real_name

    return nickname_map

def apply_substitutions(log_lines, substitutions):
    sorted_subs = sorted(substitutions.items(), key=lambda x: len(x[0]), reverse=True)
    result = []
    for line in log_lines:
        new_line = []
        for el in line:
            for old, new in sorted_subs:
                el = re.sub(r'(?<![A-Za-z0-9\-])' + re.escape(old) + r'(?![A-Za-z0-9\-])', str(new), el)
            new_line.append(el)
        result.append(new_line)
    return result

# ─── raccolta info ────────────────────────────────────────────

def parse_battle_log(log_lines):
    """
    Estrae per ogni pokemon: abilità, item, mosse usate.
    La chiave del dizionario è 'pX_POKEID' (es. 'p0_0001').
    """
    info = defaultdict(lambda: {'ability': None, 'item': None, 'moves': set()})
    ability_id ={}
    item_id = {}
    for line in log_lines:
        if not line:
            continue
        tag = line[0]

        # ── abilità esplicita ──────────────────────────────
        # ['-ability', 'p0a: 0001', 'Fairy Aura']
        if tag == '-ability' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                if line[2] not in ability_id.keys():
                    ability_id[line[2]] = get_ability_id(line[2])
                info[f'p{player}_{poke_id}']['ability'] = ability_id[line[2]]


        # ── abilità da [from] ability: ────────────────────
        # appare in righe tipo ['-weather', ..., '[from] ability: Sand Stream', '[of] p0a: 0001']
        for field in line:
            if '[from] ability:' in field:
                ability_name = field.split('[from] ability:')[1].strip()
                of_field = next((f for f in line if '[of]' in f), None)
                if of_field:
                    of_clean = of_field.replace('[of] ', '').strip()
                    ref = extract_pokemon_id(of_clean)
                    if ref:
                        player, _, poke_id = ref
                        if ability_name not in ability_id.keys():
                            ability_id[ability_name] = get_ability_id(ability_name)
                        info[f'p{player}_{poke_id}']['ability'] = ability_id[ability_name]

        # ── mosse ─────────────────────────────────────────
        # ['move', 'p0a: 0001', 'Fake Out', ...]
        if tag == 'move' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                info[f'p{player}_{poke_id}']['moves'].add(Move(line[2],0,0,0,0).id)

        # ── item consumato ────────────────────────────────
        # ['-enditem', 'p2a: 0003', 'White Herb']
        if tag == '-enditem' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                if line[2] not in item_id.keys():
                    item_id[line[2]] = get_item_id(line[2])
                info[f'p{player}_{poke_id}']['item'] = item_id[line[2]]



        # ── item da [from] item: ──────────────────────────
        # ['-heal', 'p0a: 0001', '64/100', '[from] item: Leftovers']
        for field in line:
            if '[from] item:' in field:
                item_name = field.split('[from] item:')[1].strip()
                ref = extract_pokemon_id(line[1]) if len(line) > 1 else None
                if ref:
                    player, _, poke_id = ref
                    info[f'p{player}_{poke_id}']['item'] = item_name
                    if item_name not in item_id.keys():
                        item_id[item_name] = get_item_id(item_name)
                    info[f'p{player}_{poke_id}']['item'] = item_id[item_name]

    return info, ability_id, item_id

# ─── entry point ──────────────────────────────────────────────

def get_slot(s):
    if s == 'a' or s == 1: return 1
    return 2

def update_pokemon(mons, info):
    for m in mons:
        if m.player == 0:
            for k, data in info.items():
                if k[1] == '0' and int(k[3:]) == m.poke_id:
                    keys = data.keys()
                    if data['ability'] != None:
                        m.ability = data['ability']
                    if data['item'] != None:
                        m.item = data['item']
                    if 'moves' in keys:
                        for mo in data['moves']:
                            m.add_move(mo)               

def convert_log(raw_lines):
    nickname_map = build_nickname_map(raw_lines)
    #print('nickname map:', nickname_map)
    raw_lines = [replace_in_line(raw, nickname_map) for raw in raw_lines]

    log_lines = filter_lines(raw_lines)

    substitutions = {}  # niente hardcoding
    mons = []
    pre_sub = { }
    mons = []
    for line in [l for l in log_lines if l[0] == 'poke']:
        name = line[2].split(',')[0].strip()
        pkmn = Pokemon(int(line[1][1])-1, name)
        substitutions[name] = str(pkmn.poke_id)
        mons.append(pkmn)
    
    ALIASES = {
    'Floette-Eternal': str(substitutions.get('Floette-Eternal', '10061')),
    'Floette': str(substitutions.get('Floette-Eternal', '10061')),
}
    substitutions.update(ALIASES)

    megas = []
    for line in log_lines:
        tag = line[0]
        if tag == 'detailschange':
            player = int(line[1][1])-1
            pok = Pokemon(player,line[2].split(',')[0])
            megas.append(pok)
            pre_sub[pok.name] = str(pok.poke_id)


    log_lines = apply_substitutions(log_lines, pre_sub)

    log_lines = apply_substitutions(log_lines, substitutions)

    info, abilities, items = parse_battle_log(log_lines)
    update_pokemon(mons,info)#usa le info che hai ottenuto per aggiornare i dati sui mons
    turn = 1
    tokens = []
    turn_moves = []
    field = Battlefield(1,int(log_lines[-1][-1]))

    for line in log_lines:
        if not line:
            continue
        tag = line[0]

        if tag == 'switch':

            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            poke_name = line[2].split(',')[0] #get entering info

            present = [p for p in mons if p.player == player and p.slot == slot] #check existent placed
            if len(present)>0: 
                present[0].slot = 0

            target = [p for p in mons if p.player == player and p.poke_id == int(poke_name)][0] 
            target.seen = 1
            turn_moves.append(Move(-1,target.player,target.slot,player,slot))
            #replace poke
            target.slot = slot

        elif tag == 'detailschange':
            player, slot = int(line[1][1])-1, get_slot(line[1][2])
            megaP = [m for m in megas if m.poke_id == int(line[2].split(',')[0]) and m.player == player][0]
            for i in range(len(mons)):
                if mons[i].player == player and mons[i].slot == slot:
                    mons[i].poke_id = megaP.poke_id
                    mons[i].stats = megaP.stats[:]
                    mons[i].types = megaP.types[:]
                    mons[i].ability = megaP.ability
                    break

        elif tag == '-sidestart':
            if 'Tailwind' in line[2]:
                player = int(line[1][-1])
                field.speed_modifier.set_bit(player,1)

        elif tag == '-sideend':
            if 'Tailwind' in line[2]:
                player = int(line[1][-1])
                field.speed_modifier.set_bit(player,0)

        elif tag == 'move':
            m = Move(line[2],int(line[1][1])-1, get_slot(line[1][2]),int(line[1][1])-1, get_slot(line[1][2]))
            tm = [p for p in mons if p.player == int(line[1][1])-1 and p.slot == get_slot(line[1][2])][0] 
            tm.add_move(m.id)


            if(line[-1]) == '[still]': #move has failed or it gets a turn to loading
                turn_moves.append(m)
            else:
                turn_moves.append(Move(line[2],int(line[1][1])-1, get_slot(line[1][2]),int(line[3][1])-1, get_slot(line[3][2])))
        
        elif tag == '-damage' or tag == '-heal':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            hp_res = float(re.split(r'[/, ]', line[2])[0])
            tm = [p for p in mons if p.player == player and p.slot == slot][0] 
            
            tm.hp_ratio = hp_res/100

        elif tag == '-enditem':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            item = line[2]
            if player == 1:
                tm = [p for p in mons if p.player == player and p.slot == slot][0] 
                tm.item = items[item]

        elif tag == '-boost':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            stat_name = line[2]
            val = int(line[3])
            tm = [p for p in mons if p.player == player and p.slot == slot][0] 
            tm.stats_change[stat_code[stat_name]]+val

        elif tag == '-unboost':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            stat_name = line[2]
            val = int(line[3])
            tm = [p for p in mons if p.player == player and p.slot == slot][0] 
            tm.stats_change[stat_code[stat_name]]-val

        elif tag == '-status':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            status = line[2]
            tm = [p for p in mons if p.player == player and p.slot == slot][0] 
            tm.status.set_bit(all_status[status],1)
        
        elif tag == '-curestatus':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            slot = get_slot(line[1][2])           # 'a' o 'b'
            status = line[2]
            tm = [p for p in mons if p.player == player and p.slot == slot][0] 
            tm.status.set_bit(all_status[status],0)

        elif tag == '-ability':
            player = int(line[1][1]) - 1   # p1→0, p2→1
            if player == 1:
                slot = get_slot(line[1][2])           # 'a' o 'b'
                abil = line[2]
                tm = [p for p in mons if p.player == player and p.slot == slot][0] 
                tm.ability = abilities[abil]

        elif tag == '-weather':
            field.current_weather = weather[line[1]]

        elif tag == 'turn':
            if int(line[1]) >= turn:
                token = []
                while len(turn_moves)<4:
                    turn_moves.append(Move(0,0,0,0,0)) #padder for the final rounds when few pokemons remain
                for mo in turn_moves[:4]:
                    token += mo.to_list()
                for poke in mons:
                    token+=poke.to_list()
                token += field.to_list()
                tokens.append(token)
                turn_moves = []
                field.turn+=1
                
        
        elif tag == 'win':
            token = []
            while len(turn_moves)<4:
                    turn_moves.append(Move(0,0,0,0,0))
            for mo in turn_moves[:4]:
                token += mo.to_list()
            for poke in mons:
                token+=poke.to_list()
            token += field.to_list()
            tokens.append(token)
            turn_moves = []
            field.turn+=1
        else: 
            pass

    return tokens



if __name__ == "__main__":
    existent_logs = os.listdir("../logs/")
    for logfile in existent_logs[1:2]:
        print(logfile)
        with open("../logs/" + logfile) as f:
            raw = f.read().split('\n')
        toks = convert_log(raw)
        for t in toks:
            print(len(t))
        
        save_to_csv(toks, "../csv/"+logfile.split('.')[0]+".csv", fill_none=0)
    
