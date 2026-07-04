from getter import *
import os
import re
from collections import defaultdict
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
 
# ─── dtype strutturati ───────────────────────────────────────
# Rispecchiano esattamente i to_list() delle classi in getter.py
#
# Move.to_list()      → [id, type, d_class, t_class, accuracy, power, priority]  (7)
# Pokemon.to_list()   → [player, slot, poke_id,            (3)
#                         type1, type2,                     (2)
#                         ability, item,                    (2)
#                         hp, atk, def, spa, spd, spe,     (6)
#                         atk_c, def_c, spa_c, spd_c, spe_c,(5)
#                         status_mask,                      (1)
#                         move0…move3 (6×4),                (24)
#                         hp_ratio]                         (1)  → totale 44
# Battlefield.to_list()→ [turn, weather, speed_mask, winner] (4)
# Action.to_list()    → [usr_pl, usr_slot, trg_pl, trg_slot, move, mega] (6)
# Turno totale        → 12×44 + 4 + 2×6 = 544
 
_MOVE_DT = np.dtype([
    ('id',     np.int16),
    ('type',     np.int16),
    ('d_class',  np.int8),
    ('t_class',  np.int8),
    ('accuracy', np.int16),
    ('power',    np.int16),
    ('priority', np.int8),
])
 
_POKEMON_DT = np.dtype([
    ('player',      np.int8),
    ('slot',        np.int8),
    ('poke_id',     np.int16),
    ('type1',       np.int8),
    ('type2',       np.int8),
    ('ability',     np.int16),
    ('item',        np.int16),
    ('hp_base',     np.int16),
    ('atk',         np.int16),
    ('def_',        np.int16),
    ('spa',         np.int16),
    ('spd',         np.int16),
    ('spe',         np.int16),
    ('atk_c',       np.int8),
    ('def_c',       np.int8),
    ('spa_c',       np.int8),
    ('spd_c',       np.int8),
    ('spe_c',       np.int8),
    ('status_mask', np.int8),   # bitmask a 6 bit, salvata come intero
    ('move0',       _MOVE_DT),
    ('move1',       _MOVE_DT),
    ('move2',       _MOVE_DT),
    ('move3',       _MOVE_DT),
    ('hp_ratio',    np.float32),
])
 
_FIELD_DT = np.dtype([
    ('turn',       np.int16),
    ('weather',    np.int8),
    ('speed_mask', np.int8),     # bitmask a 3 bit, salvata come intero
    ('winner',     np.int8),
])
 
_ACTION_DT = np.dtype([
    ('usr_pl',   np.int8),
    ('usr_slot', np.int8),
    ('trg_pl',   np.int8),
    ('trg_slot', np.int8),
    ('move',     np.int8),
    ('mega',     np.int8),
])
 
_TURN_DT = np.dtype([
    ('pokemon', _POKEMON_DT, (12,)),
    ('field',   _FIELD_DT),
    ('action0', _ACTION_DT),
    ('action1', _ACTION_DT),
])
 
 
def _token_to_structured(token):
    """
    Converte una lista piatta (output di _build_token) in un record numpy strutturato.
    Struttura attesa: [pokemon x12 (44 campi ciascuno), field x4, action x2 (6 campi ciascuno)] tot = 544
    """
    record = np.zeros(1, dtype=_TURN_DT)[0]
 
    offset = 0
    POKE_SIZE = 48
    MOVE_SIZE = 7
 
    for i in range(12):
        p = token[offset: offset + POKE_SIZE]
        pk = record['pokemon'][i]
        pk['player']      = p[0]
        pk['slot']        = p[1]
        pk['poke_id']     = p[2]
        pk['type1']       = p[3]
        pk['type2']       = p[4]
        pk['ability']     = p[5]
        pk['item']        = p[6]
        pk['hp_base']     = p[7]
        pk['atk']         = p[8]
        pk['def_']        = p[9]
        pk['spa']         = p[10]
        pk['spd']         = p[11]
        pk['spe']         = p[12]
        pk['atk_c']       = p[13]
        pk['def_c']       = p[14]
        pk['spa_c']       = p[15]
        pk['spd_c']       = p[16]
        pk['spe_c']       = p[17]
        pk['status_mask'] = p[18]
        for j, mname in enumerate(('move0', 'move1', 'move2', 'move3')):
            base = 19 + j * MOVE_SIZE
            m = p[base: base + MOVE_SIZE]
            pk[mname]['id']     = m[0]
            pk[mname]['type']     = m[1]
            pk[mname]['d_class']  = m[2]
            pk[mname]['t_class']  = m[3]
            pk[mname]['accuracy'] = m[4]
            pk[mname]['power']    = m[5]
            pk[mname]['priority'] = m[6]
        pk['hp_ratio'] = p[47]
        offset += POKE_SIZE
 
    # field (4 campi)
    record['field']['turn']       = token[offset]
    record['field']['weather']    = token[offset + 1]
    record['field']['speed_mask'] = token[offset + 2]
    record['field']['winner']     = token[offset + 3]
    offset += 4
 
    # 2 azioni (6 campi ciascuna)
    for aname in ('action0', 'action1'):
        a = token[offset: offset + 6]
        record[aname]['usr_pl']   = a[0]
        record[aname]['usr_slot'] = a[1]
        record[aname]['trg_pl']   = a[2]
        record[aname]['trg_slot'] = a[3]
        record[aname]['move']     = a[4]
        record[aname]['mega']     = a[5]
        offset += 6
 
    return record
 
 
def save_to_npz(tokens, filename):
    """
    Salva i token di una partita in un file .npz con dtype strutturato e compressione.
    Ogni elemento di tokens è un turno; il file contiene un array di shape (T,).
    Caricamento: data = np.load(file); turns = data['turns']
    """
    records = np.array(
        [_token_to_structured(t) for t in tokens],
        dtype=_TURN_DT
    )
    np.savez_compressed(filename, turns=records)
 

# ─── costanti ────────────────────────────────────────────────
INTERESTING_TAGS = {
    "poke", "-ability", "turn", "move", "-damage", "detailschange","switch",
    "-sidestart", "-sideend", "-enditem", "-weather", "win",
    "-heal", "-boost", "-unboost", "-status", "-start", "-fieldstart", "-fieldend","drag","replace"
}

# ─── parsing ─────────────────────────────────────────────────

def parse_field(field):
    """Splitta un campo su ':' o ',' restituendo una lista di token puliti."""
    tokens = re.split(r'[:,]', field)
    return [t.strip() for t in tokens if t.strip()]

def filter_lines(raw_lines):
    if len(raw_lines) < 2:
        return []
    names = []
    i = 0
    for l in raw_lines:
        if l.startswith('|j|'):
            names.append(l[4:]) 
            i+=1
        if i == 2: 
            break
    p1_name = names[0]
    p2_name = names[1]

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

def apply_mega_subs(line, active_mega_subs):
    """Applica le sostituzioni mega attive a una riga già processata, solo per il player corretto."""
    if not any(active_mega_subs.values()):
        return line
    
    result = []
    for el in line:
        new_el = el
        for player_idx, subs in active_mega_subs.items():
            player_str = f"p{player_idx + 1}" # Crea "p1" o "p2"
            for old_val, new_id in subs.items():
                if old_val.isdigit():
                    # Se è un ID base (es. 3), sostituisci solo se preceduto dal prefisso slot corretto (es p1a:)
                    pattern = r'(' + player_str + r'[a-c]:\s*)' + re.escape(old_val) + r'(?!\d)'
                    new_el = re.sub(pattern, r'\g<1>' + new_id, new_el)
                else:
                    # Se è il nome testuale della Mega (es. "Venusaur-Mega"), sostituiscilo direttamente
                    pattern = r'(?<![A-Za-z0-9\-])' + re.escape(old_val) + r'(?![A-Za-z0-9\-])'
                    new_el = re.sub(pattern, new_id, new_el)
        result.append(new_el)
    return result

# ─── sostituzione nomi → id ───────────────────────────────────

def replace_in_line(raw, substitutions):
    sorted_subs = sorted(substitutions.items(), key=lambda x: len(x[0]), reverse=True)
    for old, new in sorted_subs:
        raw = re.sub(r'(?<![A-Za-z0-9\-])' + re.escape(old) + r'(?![A-Za-z0-9\-])', str(new), raw)
    return raw

def build_nickname_map(raw_lines):
    nickname_map = {}
    for raw in raw_lines:
        parts = raw.split('|')
        # I nickname in Showdown emergono nei log quando il Pokémon entra in campo
        if len(parts) >= 4 and parts[1] in ('switch', 'drag', 'replace'):
            player_and_nick = parts[2] # Es: "p2b: Bulby"
            real_name = parts[3].split(',')[0].strip() # Es: "Venusaur-Mega" o "Venusaur"
            
            if ': ' in player_and_nick:
                nick = player_and_nick.split(': ')[1].strip()
                
                # Se il nickname è diverso dal nome, NON è una Mega, e non lo abbiamo ancora salvato:
                if nick != real_name and '-Mega' not in real_name and nick not in nickname_map:
                    nickname_map[nick] = real_name
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

def fix_basculegion_raw(raw_lines):
    """
    Trova i giocatori che possiedono Basculegion-F e corregge tutti 
    i riferimenti successivi nei log (che Showdown tronca a 'Basculegion')
    ripristinando 'Basculegion-F' a priori.
    """
    female_players = []
    for line in raw_lines:
        parts = line.split('|')
        # Cerca la definizione del team (es: |poke|p1|Basculegion-F, L50, F|)
        if len(parts) >= 4 and parts[1] == 'poke':
            if 'Basculegion-F' in parts[3]:
                female_players.append(parts[2]) # Salva 'p1' o 'p2'
                
    if not female_players:
        return raw_lines
        
    new_raw = []
    for line in raw_lines:
        new_line = line
        for p_str in female_players:
            # 1. Sistema gli identificatori (es: p1a: Basculegion -> p1a: Basculegion-F)
            # Il negative lookahead (?!\-F) evita di aggiungere "-F" se c'è già
            pattern_id = r'(' + p_str + r'[a-c]:\s*)Basculegion(?!\-F)(?![A-Za-z0-9\-])'
            new_line = re.sub(pattern_id, r'\g<1>Basculegion-F', new_line)
            
            # 2. Sistema i nomi della specie puramente testuali negli eventi di entrata
            # (es: |switch|p1a: Basculegion-F|Basculegion, L50... -> |switch|...|Basculegion-F, L50...)
            if new_line.startswith(f"|switch|{p_str}") or new_line.startswith(f"|drag|{p_str}") or new_line.startswith(f"|replace|{p_str}"):
                new_line = re.sub(r'\|Basculegion(?!\-F)', '|Basculegion-F', new_line)
                
        new_raw.append(new_line)
    return new_raw

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
    #stri = '' #just for debug

    for poke in mons:
        #stri+=str(poke.name) + ', '
        token += poke.to_list()  # include internamente le 4 mosse conosciute

    # campo
    token += field.to_list()
    #stri+= str(field.to_list())+ ', '

    # azioni player 0: esattamente 2 (padding con Action vuota se necessario)
    p0_actions = turn_actions[:2]
    while len(p0_actions) < 2:
        p0_actions.append(Action(0, 0, 0, 0, 5))  # 5 = azione sconosciuta/padding

    for action in p0_actions:
        #stri+=str(action)+', '
        token += action.to_list()
    #print(stri)
    return token

#───────────────────────────────────────────────────────────────
# ─── Entry point ──────────────────────────────────────────────
#───────────────────────────────────────────────────────────────
def convert_log(raw_lines):
    raw_lines = fix_basculegion_raw(raw_lines)
    nickname_map = build_nickname_map(raw_lines)
    raw_lines = [replace_in_line(raw, nickname_map) for raw in raw_lines]
    log_lines = filter_lines(raw_lines)
    
    active_mega_subs = {0: {}, 1: {}}
    substitutions = {}
    mons = []

    # ── costruzione team iniziale (forme base) ────────────────
    for line in [l for l in log_lines if l[0] == 'poke']:
        name = line[2].split(',')[0].strip()
        pkmn = Pokemon(int(line[1][1]) - 1, name)
        substitutions[name] = str(pkmn.poke_id)
        mons.append(pkmn)

    ALIASES = {
        'Floette-Eternal': str(substitutions.get('Floette-Eternal', '10061')),
        'Floette': str(substitutions.get('Floette-Eternal', '10061')),
        'Zoroark-Hisui': str(substitutions.get('Zoroark-Hisui', '10239')),
        'Zoroark': str(substitutions.get('Zoroark-Hisui', '10239')),
    }
    substitutions.update(ALIASES)

    # ── NOVITÀ: Raccolta forme alternative, escludendo le Mega ──
    for line in log_lines:
        if line[0] in ('switch', 'drag', 'replace'):
            name = line[2].split(',')[0].strip()
            # Aggiunge Hisui, Galar, ecc., ma lascia inalterati i nomi con "-Mega"
            if '-Mega' not in name and name not in substitutions:
                pkmn = Pokemon(0, name) 
                substitutions[name] = str(pkmn.poke_id)

    log_lines = apply_substitutions(log_lines, substitutions)
    info, abilities, items = parse_battle_log(log_lines)
    update_pokemon(mons, info)

    tokens = []
    turn_actions = []   
    winner = int(log_lines[-1][-1])
    
    field = Battlefield(0, winner)
    mega_used = [[0,0],[0,0]]

    for line in log_lines:
        if not line:
            continue
        
        # Le mega si applicano da qui in poi per le righe successive al detailschange
        line = apply_mega_subs(line, active_mega_subs)
        tag = line[0]

# ── switch, drag e replace ────────────────────────────────────
        if tag in ('switch', 'drag', 'replace'):

            player = int(line[1][1]) - 1
            field_slot = get_slot(line[1][2])   # 1 o 2 (posizione in campo)
            poke_name = line[2].split(',')[0]
            # for p in mons:
            #     print(f'{p.poke_id}, {p.player}, {p.slot}')
            # print(line)
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
            target.slot = field_slot

            # Registriamo l'azione solo per player 0 e se è uno switch volontario
            if player == 0 and tag == 'switch':
                turn_actions.append(Action(
                    usr_pl=0,
                    usr_slot=field_slot,
                    trg_pl=0,
                    trg_slot=incoming_slot,
                    move=4
                ))

        # ── detailschange (mega/forma) ────────────────────────
        elif tag == 'detailschange':
            player, slot = int(line[1][1]) - 1, get_slot(line[1][2])
            mega_used[player][slot-1] = 1
            
            # Il nome qui sarà testo puro (es. "Venusaur-Mega") perché l'abbiamo escluso dal ciclo in alto
            mega_name = line[2].split(',')[0].strip()
            megaP = Pokemon(player, mega_name)

            # Trova l'id base del pokemon
            poke_in_slot = next(p for p in mons if p.player == player and p.slot == slot)
            base_id = str(poke_in_slot.poke_id)   
            mega_id_str = str(megaP.poke_id)

            # Innesca la sostituzione per tutte le ITERAZIONI FUTURE del ciclo:
            # 1. Sostituisce i puntatori posizionali di Showdown (es. "p1a: 3" diventa "p1a: 10033")
            active_mega_subs[player][base_id] = mega_id_str
            # 2. Sostituisce il testo in caso di switch futuri dalla panchina (es. "Venusaur-Mega" diventa "10033")
            active_mega_subs[player][mega_name] = mega_id_str

            # Aggiorna l'oggetto
            poke_in_slot.poke_id = megaP.poke_id
            poke_in_slot.stats   = megaP.stats[:]
            poke_in_slot.types   = megaP.types[:]
            poke_in_slot.ability = megaP.ability

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
                    5 # 5 = mossa sconosciuta/padding
                )
                if '[still]' in line:
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
    #gen9championsvgc2026regma-2629670048.txt

    existent_logs = os.listdir("../logs/")
    oc_poke,oc_move,oc_item,oc_abilities = get_cache_stats()
    existent_npz = os.listdir("../npz/")
    for logfile in tqdm(existent_logs[:]):
        #logfile = 'Rgen9championsvgc2026regma-2626479231.txt'
        print(logfile)
        # if logfile.split('.')[0]+'.npz' in existent_npz:
        #     print('present')
        #     continue

        with open("../logs/" + logfile) as f:
            raw = f.read()
            if "Aurora|Froslass" in raw:
                print('skipped')
                continue
            raw = raw.split('\n')

        toks = convert_log(raw)
        c_poke,c_move,c_item,c_abilities = get_cache_stats()
        print(f'turns: {len(toks)}, poke: {c_poke}(+{c_poke-oc_poke}), moves: {c_move}(+{c_move-oc_move}), items: {c_item}(+{c_item-oc_item}), abilities: {c_abilities}(+{c_abilities-oc_abilities})')
        oc_poke,oc_move,oc_item,oc_abilities = c_poke,c_move,c_item,c_abilities 
        save_to_npz(toks, "../npz/" + logfile.split('.')[0])
