from Model.PokemonVGCDataset import PokemonVGCDataset
from torch.utils.data import DataLoader # type: ignore
from sklearn.model_selection import train_test_split # type: ignore



# 4. Passalo alla funzione di train che abbiamo scritto prima
# train_decision_transformer(model, dataloader, num_epochs, device)
def main():
    # 1. Configurazione del Device (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto addestrando sul device: {device}")

    # 2. Preparazione dei Dati (Dataset e DataLoader)

    # 1. Trova tutti i file .npy nella cartella dei dati
    lista_files = glob.glob("/kaggle/input/datasets/armandocoppola/reg-m-a/reg_m-A/*.npz")

    #Splittiamo i dati in train, validation e test (80% train, 10% validation, 10% test)
    train_files, temp_files = train_test_split(lista_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # 2. Inizializza il Dataset
    dataset_training = PokemonVGCDataset(file_paths=train_files, max_turn=49)
    dataset_validation = PokemonVGCDataset(file_paths=val_files, max_turn=49)
    dataset_test = PokemonVGCDataset(file_paths=test_files, max_turn=49)

    # 3. Crea il DataLoader del training set
    # num_workers velocizza il caricamento parallelizzando la lettura su CPU
    dataloader_training = DataLoader(
        dataset_training,
        batch_size=32, 
        shuffle=True, 
        num_workers=4, 
        drop_last=True
    )

    #Creiamo il DataLoader del validation set
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )

    # Creiamo il DataLoader del test set
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )

    # 3. Inizializzazione del Modello
    # Richiamiamo il modello usando le dimensioni che avete definito
    model = DecisionTransformer(
        action_dim=192*2, 
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
        dataloader_training=dataloader_training, # Passiamo il raggruppatore di dati
        dataloader_validation=dataloader_validation,
        num_epochs=EPOCHS, 
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
