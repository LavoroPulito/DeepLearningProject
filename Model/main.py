import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from PokemonVGCDataset import *
from TrainingLoop import *

# Importate le vostre classi (assumendo che siano in file separati)
from DecisionTransformer import DecisionTransformer
# from data_module import PokemonVGCDataset # Da creare: la classe che gestisce i vostri dati
import glob

# 1. Trova tutti i file .npy nella cartella dei dati
lista_files = glob.glob("../npz/reg_m-A/*.npz")

# 2. Inizializza il Dataset
dataset = PokemonVGCDataset(file_paths=lista_files, max_turn=49)

# 3. Crea il DataLoader
# num_workers velocizza il caricamento parallelizzando la lettura su CPU
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4, 
    drop_last=True
)

# 4. Passalo alla funzione di train che abbiamo scritto prima
# train_decision_transformer(model, dataloader, epochs, device)
def main():
    # 1. Configurazione del Device (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto addestrando sul device: {device}")

    # 2. Preparazione dei Dati (Dataset e DataLoader)
    # Qui dovrete istanziare la vostra classe Dataset personalizzata che legge i log delle partite VGC
    # dataset = PokemonVGCDataset(percorso_file="dati_vgc.json")
    
    # Il DataLoader si occupa di pescare i dati dal dataset e raggrupparli in batch
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Inizializzazione del Modello
    # Richiamiamo il modello usando le dimensioni che avete definito
    model = DecisionTransformer(
        action_dim=192, 
        d_model=256, 
        n_heads=8, 
        depth=6, 
        max_turn=49
    )

    # 4. Definizione degli Iperparametri
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    print("Avvio dell'addestramento...")
    
    # 5. LA CHIAMATA ALLA FUNZIONE DI TRAINING
    train_decision_transformer(
        model=model, 
        dataloader=dataloader, # Passiamo il raggruppatore di dati
        epochs=EPOCHS, 
        device=device, 
        lr=LEARNING_RATE
    )
    
    print("Addestramento completato!")

    # 6. Salvataggio del modello addestrato
    # Salviamo solo i "pesi" (state_dict) per poterlo ricaricare in fase di inferenza/gioco
    torch.save(model.state_dict(), "vgc_decision_transformer.pth")
    print("Modello salvato con successo.")

if __name__ == "__main__":
    main()
