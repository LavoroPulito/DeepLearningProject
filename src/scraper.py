import requests # type: ignore
from tqdm import tqdm # type: ignore
import os
# 1. Inserisci l'URL della pagina o dell'API
replayUrl = "https://replay.pokemonshowdown.com/" #base for download replay. add '.json' at the end after append the battleID
battleHistoryUrl = "https://replay.pokemonshowdown.com/search.json?format=[Gen%209%20Champions]%20VGC%202026%20Reg%20M-B&sort=rating&page="

def scarica_id(targetUrl):
    ids = []
    try:
        # 2. Fai la richiesta GET alla pagina
        risposta = requests.get(targetUrl)
    
        # 3. Controlla che la richiesta sia andata a buon fine (codice 200)
        risposta.raise_for_status() 
    
        # 4. Estrai il JSON e convertilo in un dizionario Python
        dati_json = risposta.json()
    
    # 5. Ora puoi usare i dati!
        print(len(dati_json)," logs scaricati con successo:")
#        print(dati_json[0].keys())
        for dic in dati_json:
            ids.append(dic['id'])
    
        # Esempio: accedere a un valore specifico
        # print("Il titolo è:", dati_json['title'])

    except requests.exceptions.RequestException as e:
        print(f"Si è verificato un errore di connessione: {e}")
    except ValueError:
        print("La pagina non ha restituito un JSON valido.")
    return ids

def scarica_log(battleID):
    try:
        # 2. Fai la richiesta GET alla pagina
        risposta = requests.get(replayUrl+battleID+'.json')
        
    
        # 3. Controlla che la richiesta sia andata a buon fine (codice 200)
        risposta.raise_for_status() 
    
        # 4. Estrai il JSON e convertilo in un dizionario Python
        dati_json = risposta.json()
    
        # 5. Ora puoi usare i dati!
        #print("Dati scaricati con successo:")
        log_file = open("../logs/"+battleID+'.txt','w')
        log_file.write(dati_json['log'])
        log_file.close()
    
    except requests.exceptions.RequestException as e:
        print(f"Si è verificato un errore di connessione: {e}")
    except ValueError:
        print("La pagina non ha restituito un JSON valido.")

def scambia_giocatori_file(percorso_file):
    # 1. Legge il contenuto del file originale
    with open(percorso_file, 'r', encoding='utf-8') as file:
        contenuto = file.read()
    
    # 2. Scambia 'p1' con 'p2' usando un placeholder temporaneo
    # Scegliamo un placeholder improbabile da trovare nel testo
    contenuto = contenuto.replace('p1', '@@TEMP_P1@@')
    contenuto = contenuto.replace('p2', 'p1')
    contenuto = contenuto.replace('@@TEMP_P1@@', 'p2')
    
    # 3. Costruisce il nuovo nome del file aggiungendo 'R' all'inizio
    cartella, nome_file = os.path.split(percorso_file)
    nuovo_nome_file = 'R' + nome_file
    nuovo_percorso = os.path.join(cartella, nuovo_nome_file)
    
    # 4. Scrive il nuovo contenuto nel nuovo file
    with open(nuovo_percorso, 'w', encoding='utf-8') as nuovo_file:
        nuovo_file.write(contenuto)

if __name__ == '__main__':
    for i in range(100):
        print('page ',i)
        ids = scarica_id(battleHistoryUrl+str(i))
        existent_logs = os.listdir("../logs/")
        existent_logs = [f[:-4] for f in existent_logs]  # rimuove .txt, più pythonico
        existent_logs_set = set(existent_logs)  # set per ricerca O(1) invece di O(n)
        ids_to_download = [Id for Id in ids if Id not in existent_logs_set]
        esistenti = len(ids) - len(ids_to_download)

        #print("ne esistevano ",esistenti, "ne scaricherai ", len(ids_to_download) )
        for ID in ids_to_download:
            scarica_log(ID)
    # existent_logs = os.listdir("../logs/")
    # for logfile in tqdm(existent_logs[:]):
    #     scambia_giocatori_file("../logs/"+logfile)


    



