from getter import *
import os
import re
from collections import defaultdict

# ─── costanti ────────────────────────────────────────────────
INTERESTING_TAGS = {
    "poke", "-ability", "turn", "move", "-damage", "detailschange",
    "-sidestart", "-sideend", "-enditem", "-weather", "win",
    "-heal", "-boost", "-unboost", "-status", "-start"
}

# ─── parsing ─────────────────────────────────────────────────

def parse_field(field):
    """Splitta un campo su ':' o ',' restituendo una lista di token puliti."""
    tokens = re.split(r'[:,]', field)
    return [t.strip() for t in tokens if t.strip()]

def filter_lines(raw_lines):
    """
    Riceve le righe grezze del log.
    - Sostituisce i nomi dei player con '0' e '1'
    - Sostituisce 'p1' con 'p0' per coerenza con i player
    - Splitta su '|' e tiene solo i tag interessanti
    - Restituisce lista di liste di stringhe
    """
    if len(raw_lines) < 2:
        return []

    p1_name = raw_lines[0][4:]  # "player|p1|Nome" → "Nome"
    p2_name = raw_lines[1][4:]

    result = []
    for raw in raw_lines:
        raw = raw.replace(p1_name, "0").replace(p2_name, "1")
        raw = raw.replace("p1", "p0")  # p1→p0, p2 resta p2 → 0-indexed
        parts = raw.split('|')
        if len(parts) >= 2 and parts[1] in INTERESTING_TAGS:
            result.append(parts[1:])  # togli il primo elemento vuoto prima del tag

    return result

def extract_pokemon_id(field):
    """
    Da 'p0a: 0001' o 'p2b: 0002' estrae (player_idx, position, poke_id).
    Restituisce None se il campo non è un riferimento a un pokemon.
    """
    match = re.match(r'p(\d)([ab]):\s*(.+)', field)
    if match:
        player = int(match.group(1))
        position = match.group(2)   # 'a' o 'b'
        name = match.group(3).strip()
        return player, position, name
    return None

# ─── sostituzione nomi → id ───────────────────────────────────

def apply_substitutions(log_lines, substitutions):
    """Sostituisce ogni occorrenza dei nomi con i rispettivi ID in tutto il log."""
    result = []
    for line in log_lines:
        new_line = []
        for el in line:
            for old, new in substitutions.items():
                if not isinstance(new,str):
                    el = el.replace(old, str(new))    
                else: 
                    el = el.replace(old, new)    
                 
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
    moves_seen = {}
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
                if line[2] not in moves_seen.keys():
                    moves_seen[line[2]] = Move(line[2],0,0)
                info[f'p{player}_{poke_id}']['moves'].add(moves_seen[line[2]].id)

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

    return info, ability_id, moves_seen, item_id

# ─── entry point ──────────────────────────────────────────────

def update_pokemon(mons, abilities, moves, item):
    for m in mons:
        for ka, a in abilities:
            #TODO controlla che le abilità sono indicizzate male, devi controllare che siano univoche sul pokemon. due incy uno blaze e uno intimidate
            pass

def convert_log(raw_lines):
    log_lines = filter_lines(raw_lines)

    # costruisci dizionario nome → id per le sostituzioni
    substitutions = {
        'Floette': '10061'
    }
    mons = []
    for line in [l for l in log_lines if l[0] == 'poke']:
        name = line[2].split(',')[0].strip()
        pkmn = Pokemon(int(line[1][1]) - 1, name)
        substitutions[name] = str(pkmn.poke_id)
        mons.append(pkmn)

    log_lines = apply_substitutions(log_lines, substitutions)
    info, abilities, moves, items = parse_battle_log(log_lines)

    update_pokemon()#usa le info che hai ottenuto per aggiornare i dati sui mons


    for l in log_lines: print(l)
    print(abilities,'\n',moves, '\n',items)
    
    return info, mons


if __name__ == "__main__":
    existent_logs = os.listdir("../logs/")
    for logfile in existent_logs[:1]:
        with open("../logs/" + logfile) as f:
            raw = f.read().split('\n')
        info, mons = convert_log(raw)
        for key, data in info.items():
            print(f"\n{key}")
            print(f"  ability : {data['ability'] or 'n/a'}")
            print(f"  item    : {data['item'] or 'n/a'}")
            print(f"  moves   : {', '.join(str(m) for m in sorted(data['moves'])) or 'n/a'}")   