import os
import requests # type: ignore
from tqdm import tqdm # type: ignore

# --- CONFIGURATION & CONSTANTS ---
REPLAY_BASE_URL = "https://replay.pokemonshowdown.com/"
BATTLE_HISTORY_URL = "https://replay.pokemonshowdown.com/search.json?format=[Gen%209%20Champions]%20VGC%202026%20Reg%20M-B&sort=rating&page="
LOGS_DIR = "../logs/" # Consider changing this to "./data/raw/" based on the new structure

def fetch_battle_ids(target_url):
    """
    Fetches a list of battle IDs from the given Showdown search URL.
    Returns an empty list if the request fails.
    """
    ids = []
    try:
        response = requests.get(target_url)
        response.raise_for_status() 
        json_data = response.json()
        
        print(f"{len(json_data)} logs successfully fetched.")
        for entry in json_data:
            ids.append(entry['id'])

    except requests.exceptions.RequestException as e:
        print(f"Connection error occurred: {e}")
    except ValueError:
        print("The page did not return a valid JSON.")
        
    return ids

def download_log(battle_id):
    """
    Downloads the replay log for a specific battle ID and saves it as a text file.
    """
    try:
        response = requests.get(f"{REPLAY_BASE_URL}{battle_id}.json")
        response.raise_for_status() 
        json_data = response.json()
        
        # Ensure the destination directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        file_path = os.path.join(LOGS_DIR, f"{battle_id}.txt")
        with open(file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(json_data['log'])
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error occurred for {battle_id}: {e}")
    except ValueError:
        print(f"The page did not return a valid JSON for {battle_id}.")

def swap_players_in_file(file_path):
    """
    Swaps 'p1' and 'p2' references inside the log file to generate 
    a reversed perspective of the battle. Saves it with an 'R' prefix.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Temporary placeholder to avoid logic conflicts during replacement
    content = content.replace('p1', '@@TEMP_P1@@')
    content = content.replace('p2', 'p1')
    content = content.replace('@@TEMP_P1@@', 'p2')
    
    folder, file_name = os.path.split(file_path)
    new_file_name = f"R{file_name}"
    
    # If using the new structure, change 'folder' here to the processed directory path
    new_file_path = os.path.join(folder, new_file_name) 
    
    with open(new_file_path, 'w', encoding='utf-8') as new_file:
        new_file.write(content)

if __name__ == '__main__':
    # --- DOWNLOAD LOGIC (Currently Commented Out) ---
    # os.makedirs(LOGS_DIR, exist_ok=True)
    # for i in tqdm(range(100)):
    #     ids = fetch_battle_ids(BATTLE_HISTORY_URL + str(i))
    #     existent_logs = os.listdir(LOGS_DIR)
    #     
    #     # Remove .txt extension pythonically
    #     existent_logs = [f[:-4] for f in existent_logs]
    #     # Convert to set for O(1) lookup performance instead of O(n)
    #     existent_logs_set = set(existent_logs)  
    #     
    #     ids_to_download = [b_id for b_id in ids if b_id not in existent_logs_set]
    #     existing_count = len(ids) - len(ids_to_download)
    #
    #     print(f"{existing_count} logs already exist. Downloading {len(ids_to_download)} new logs.")
    #     for b_id in tqdm(ids_to_download):
    #         download_log(b_id)

    # --- SWAP PLAYERS LOGIC ---
    if os.path.exists(LOGS_DIR):
        existent_logs = os.listdir(LOGS_DIR)
        for logfile in tqdm(existent_logs):
            # Skip already reversed files to avoid double-processing
            if 'regmb' in logfile and not logfile.startswith('R'):
                file_path = os.path.join(LOGS_DIR, logfile)
                swap_players_in_file(file_path)
    else:
        print(f"The directory {LOGS_DIR} does not exist. Nothing to process.")