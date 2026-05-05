import requests
from tqdm import tqdm
# 1. Inserisci l'URL della pagina o dell'API
replayUrl = "https://replay.pokemonshowdown.com/" #base for download replay. add '.json' at the end after append the battleID
battleHistoryUrl = "https://replay.pokemonshowdown.com/search.json?format=[Gen%209%20Champions]%20VGC%202026%20Reg%20M-A&sort=rating&page=1"

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
        print("Dati scaricati con successo:")
        print(len(dati_json))
        print(dati_json[0].keys())
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
        log_file = open("logs/"+battleID+'.txt','w')
        log_file.write(dati_json['log'])
        log_file.close()
    
    except requests.exceptions.RequestException as e:
        print(f"Si è verificato un errore di connessione: {e}")
    except ValueError:
        print("La pagina non ha restituito un JSON valido.")


ids = scarica_id(battleHistoryUrl)
for ID in tqdm(ids):
    scarica_log(ID)
#TODO aggiungi un controllo sugli id già scaricati 
#TODO semplifica i nomi dei log



