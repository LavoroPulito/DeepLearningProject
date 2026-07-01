from getter import *
import os
import re
from collections import defaultdict
import copy
import csv
from tqdm import tqdm # type: ignore


def save_to_csv(tokens, filename, fill_none=0):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tokens:
            writer.writerow([fill_none if v is None else v for v in row])


# ─── costanti ────────────────────────────────────────────────
INTERESTING_TAGS = {
    "poke", "-ability", "turn", "move", "-damage", "detailschange","switch",
    "-sidestart", "-sideend", "-enditem", "-weather", "win",
    "-heal", "-boost", "-unboost", "-status", "-start", "-fieldstart", "-fieldend"
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
        raw = re.sub(r'(?<![A-Za-z0-9\-])' + re.escape(old) + r'(?![A-Za-z0-9\-])', str(new), raw)
    return raw

def build_nickname_map(raw_lines):
    nickname_map = {}
    position_to_name = {}
    for raw in raw_lines:
        parts = raw.split('|')
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == 'poke':
            player_slot = parts[2]
            real_name = parts[3].split(',')[0].strip()
            position_to_name[player_slot] = real_name

        elif tag == 'switch':
            ref = parts[2]
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
    info = defaultdict(lambda: {'ability': None, 'item': None, 'moves': set()})
    ability_id = {}
    item_id = {}
    for line in log_lines:
        if not line:
            continue
        tag = line[0]

        if tag == '-ability' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                if line[2] not in ability_id.keys():
                    ability_id[line[2]] = get_ability_id(line[2])
                info[f'p{player}_{poke_id}']['ability'] = ability_id[line[2]]

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

        if tag == 'move' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                info[f'p{player}_{poke_id}']['moves'].add(Move(line[2]))

        elif tag == '-enditem' and len(line) >= 3:
            ref = extract_pokemon_id(line[1])
            if ref:
                player, _, poke_id = ref
                if line[2] not in item_id.keys():
                    item_id[line[2]] = get_item_id(line[2])
                info[f'p{player}_{poke_id}']['item'] = item_id[line[2]]

        for field in line:
            if '[from] item:' in field:
                item_name = field.split('[from] item:')[1].strip()
                ref = extract_pokemon_id(line[1]) if len(line) > 1 else None
                if ref:
                    player, _, poke_id = ref
                    if item_name not in item_id.keys():
                        item_id[item_name] = get_item_id(item_name)
                    info[f'p{player}_{poke_id}']['item'] = item_id[item_name]

    return info, ability_id, item_id

# ─── helpers ──────────────────────────────────────────────────

def get_slot(s):
    if s == 'a' or s == 1: return 1
    return 2

def update_pokemon(mons, info):
    for m in mons:
        if m.player == 0:
            for k, data in info.items():
                if k[1] == '0' and int(k[3:]) == m.poke_id:
                    if data['ability'] is not None:
                        m.ability = data['ability']
                    if data['item'] is not None:
                        m.item = data['item']
                    if 'moves' in data.keys():
                        for mo in data['moves']:
                            m.add_move(mo)

def _build_token(mons, field, turn_actions):
    """
    Struttura token:
      - 12 pokemon ordinati per (player, slot), ognuno con le sue 4 mosse interne
      - campo di battaglia
      - 2 azioni del giocatore 0 (con padding se necessario)
    """
    token = []


    for poke in mons:
        token += poke.to_list()  # include internamente le 4 mosse conosciute

    # campo
    token += field.to_list()

    # azioni player 0: esattamente 2 (padding con Action vuota se necessario)
    p0_actions = turn_actions[:2]
    while len(p0_actions) < 2:
        p0_actions.append(Action(0, 0, 0, 0, 6))  # 6 = azione sconosciuta/padding

    for action in p0_actions:
        token += action.to_list()

    return token

# ─── entry point ──────────────────────────────────────────────

def convert_log(raw_lines):
    nickname_map = build_nickname_map(raw_lines)
    raw_lines = [replace_in_line(raw, nickname_map) for raw in raw_lines]
    log_lines = filter_lines(raw_lines)

    substitutions = {}
    mons = []

    # ── costruzione team iniziale ─────────────────────────────
    # tutti i pokemon partono con slot=0 (non ancora visti)
    for line in [l for l in log_lines if l[0] == 'poke']:
        name = line[2].split(',')[0].strip()
        print(name)
        pkmn = Pokemon(int(line[1][1]) - 1, name)
        # slot=0 di default (impostato in __init__)
        substitutions[name] = str(pkmn.poke_id)
        mons.append(pkmn)

    ALIASES = {
        'Floette-Eternal': str(substitutions.get('Floette-Eternal', '10061')),
        'Floette': str(substitutions.get('Floette-Eternal', '10061')),
    }
    substitutions.update(ALIASES)

    megas = []
    pre_sub = {}
    for line in log_lines:
        if line[0] == 'detailschange':
            player = int(line[1][1]) - 1
            pok = Pokemon(player, line[2].split(',')[0])
            megas.append(pok)
            pre_sub[pok.name] = str(pok.poke_id)

    log_lines = apply_substitutions(log_lines, pre_sub)
    log_lines = apply_substitutions(log_lines, substitutions)

    info, abilities, items = parse_battle_log(log_lines)
    update_pokemon(mons, info)

    tokens = []
    turn_actions = []   # lista di Action (solo player 0)

    #aggiungere turno 0 QUA
    
    field = Battlefield(1, int(log_lines[-1][-1]))
    mega_used = [[0,0],[0,0]]
    for line in log_lines:
        if not line:
            continue
        tag = line[0]

        # ── switch ────────────────────────────────────────────
        if tag == 'switch':
            player = int(line[1][1]) - 1
            field_slot = get_slot(line[1][2])   # 1 o 2 (posizione in campo)
            poke_name = line[2].split(',')[0]

            # pokemon entrante
            target = [p for p in mons if p.player == player and p.poke_id == int(poke_name)][0]
            incoming_slot = target.slot  # slot che aveva prima (0, 3 o 4)

            # pokemon uscente: prende lo slot dell'entrante (o primo slot panchina libero)
            present = [p for p in mons if p.player == player and p.slot == field_slot]
            if present:
                outgoing = present[0]
                if incoming_slot == 0:
                    # entrante mai visto → uscente prende primo slot panchina libero
                    bench_used = {p.slot for p in mons if p.player == player and p.slot in (3, 4)}
                    outgoing.slot = 3 if 3 not in bench_used else 4
                else:
                    # entrante viene dalla panchina → uscente prende il suo slot
                    outgoing.slot = incoming_slot

            # entrante va in campo
            target.seen = 1
            target.slot = field_slot

            # azione solo per player 0
            if player == 0:
                # indice panchina per Action: bench_slot 3→4, 4→5 (move 4 e 5 = switch)
                if incoming_slot == 0:
                    # pokemon mai visto: trova quale indice ha nel team p0
                    p0_mons = [p for p in mons if p.player == 0]
                    bench_idx = next(
                        (i for i, p in enumerate(p0_mons) if p.poke_id == target.poke_id),
                        0
                    )
                    move_code = 4 + (bench_idx % 2)  # 4 o 5
                else:
                    move_code = 4 if incoming_slot == 3 else 5
                turn_actions.append(Action(
                    usr_pl=0,
                    usr_slot=field_slot,
                    trg_pl=0,
                    trg_slot=field_slot,
                    move=move_code
                ))

        # ── detailschange (mega/forma) ────────────────────────
        elif tag == 'detailschange':
            player, slot = int(line[1][1]) - 1, get_slot(line[1][2])
            mega_used[player][slot-1] = 1
            megaP = [m for m in megas if m.poke_id == int(line[2].split(',')[0]) and m.player == player][0]
            for i in range(len(mons)):
                if mons[i].player == player and mons[i].slot == slot:
                    mons[i].poke_id = megaP.poke_id
                    mons[i].stats = megaP.stats[:]
                    mons[i].types = megaP.types[:]
                    mons[i].ability = megaP.ability
                    break

        # ── field effects ─────────────────────────────────────
        elif tag == '-sidestart':
            if 'Tailwind' in line[2]:
                player = int(line[1][-1])
                field.speed_modifier.set_bit(player, 1)

        elif tag == '-sideend':
            if 'Tailwind' in line[2]:
                player = int(line[1][-1])
                field.speed_modifier.set_bit(player, 0)

        elif tag == '-weather':
            field.current_weather = weather[line[1]]

        elif tag == '-fieldstart':
            if 'Trick Room' in line[1]:
                field.speed_modifier.set_bit(2, 1)

        elif tag == '-fieldend':
            if 'Trick Room' in line[1]:
                field.speed_modifier.set_bit(2, 0)

        # ── mosse ─────────────────────────────────────────────
        elif tag == 'move':
            player = int(line[1][1]) - 1
            usr_slot = get_slot(line[1][2])

            move_obj = Move(line[2])
            poke = [p for p in mons if p.player == player and p.slot == usr_slot][0]

            poke.add_move(move_obj)

            if player == 0:
                # trova l'indice (0-3) della mossa nel moveset
                move_slot = next(
                    (i for i, m in enumerate(poke.known_moves) if m.id == move_obj.id),
                    6  # 6 = mossa sconosciuta/padding
                )
                if line[-1] == '[still]':
                    trg_slot = usr_slot
                    trg_pl = 0
                else:
                    trg_slot = get_slot(line[3][2]) if len(line) > 3 else usr_slot
                    trg_pl = int(line[3][1]) - 1 if len(line) > 3 else 0

                turn_actions.append(Action(
                    usr_pl=0,
                    usr_slot=usr_slot,
                    trg_pl=trg_pl,
                    trg_slot=trg_slot,
                    move=move_slot,
                    mega = mega_used[player][usr_slot-1]
                ))
                mega_used[player][usr_slot-1] = 0
            

        # ── danno e cura ──────────────────────────────────────
        elif tag == '-damage' or tag == '-heal':
            player = int(line[1][1]) - 1
            slot = get_slot(line[1][2])
            hp_res = float(re.split(r'[/, ]', line[2])[0])
            tm = [p for p in mons if p.player == player and p.slot == slot][0]
            tm.hp_ratio = hp_res / 100

        # ── item consumato ────────────────────────────────────
        elif tag == '-enditem':
            player = int(line[1][1]) - 1
            slot = get_slot(line[1][2])
            item = line[2]
            if player == 1:
                tm = [p for p in mons if p.player == player and p.slot == slot][0]
                tm.item = items[item]

        # ── boost/unboost ─────────────────────────────────────
        elif tag == '-boost':
            if line[2] in stat_code.keys():
                player = int(line[1][1]) - 1
                slot = get_slot(line[1][2])
                tm = [p for p in mons if p.player == player and p.slot == slot][0]
                tm.stats_change[stat_code[line[2]]] += int(line[3])

        elif tag == '-unboost':
            if line[2] in stat_code.keys():
                player = int(line[1][1]) - 1
                slot = get_slot(line[1][2])
                tm = [p for p in mons if p.player == player and p.slot == slot][0]
                tm.stats_change[stat_code[line[2]]] -= int(line[3])

        # ── status ────────────────────────────────────────────
        elif tag == '-status':
            player = int(line[1][1]) - 1
            slot = get_slot(line[1][2])
            tm = [p for p in mons if p.player == player and p.slot == slot][0]
            tm.status.set_bit(all_status[line[2]], 1)

        elif tag == '-curestatus':
            player = int(line[1][1]) - 1
            slot = get_slot(line[1][2])
            tm = [p for p in mons if p.player == player and p.slot == slot][0]
            tm.status.set_bit(all_status[line[2]], 0)

        elif tag == '-ability':
            player = int(line[1][1]) - 1
            if player == 1:
                slot = get_slot(line[1][2])
                tm = [p for p in mons if p.player == player and p.slot == slot][0]
                tm.ability = abilities[line[2]]

        # ── fine turno ────────────────────────────────────────
        elif tag == 'turn':
            if int(line[1]) >= field.turn:
                tokens.append(_build_token(mons, field, turn_actions))
                turn_actions = []
                field.turn += 1

        elif tag == 'win':
            tokens.append(_build_token(mons, field, turn_actions))
            turn_actions = []

        for el in line: 
            if '[from] ability:' in el:
                ability_name = el.split('[from] ability:')[1].strip()
                of_field = next((f for f in line if '[of]' in f), None)
                if of_field:
                    of_clean = of_field.replace('[of] ', '').strip()
                    ref = extract_pokemon_id(of_clean)
                    if ref:
                        player, slot, poke_id = ref
                        if player == '1':
                            tm = [p for p in mons if p.player == player and p.slot == slot][0]
                            tm.ability = abilities[ability_name]

            elif '[from] item:' in el:
                item_name = el.split('[from] item:')[1].strip()
                ref = extract_pokemon_id(line[1]) if len(line) > 1 else None
                if ref:
                    player, slot, poke_id = ref
                    if player == '1':
                            tm = [p for p in mons if p.player == player and p.slot == slot][0]
                            tm.item = item[item_name]

    return tokens


if __name__ == "__main__":
    existent_logs = os.listdir("../logs/")
    ocpoke,ocmove,ocitem,ocabil = get_cache_stats()

    for logfile in tqdm(existent_logs[30:50]):
        print(logfile)
        #logfile = "gen9championsvgc2026regma-2590123586.txt"
        with open("../logs/" + logfile) as f:
            raw = f.read().split('\n')
        toks = convert_log(raw)
        cpoke,cmove,citem,cabil = get_cache_stats()
        print(f'turns: {len(toks)}, poke: {cpoke}(+{cpoke-ocpoke}), moves: {cmove}(+{cmove-ocmove}), items: {citem}(+{citem-ocitem}), abilities: {cabil}(+{cabil-ocabil})')
        ocpoke,ocmove,ocitem,ocabil = cpoke,cmove,citem,cabil 
        save_to_csv(toks, "../csv/" + logfile.split('.')[0] + ".csv", fill_none=0)
