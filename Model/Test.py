import os
import glob
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader # type: ignore


# Importa le classi e le funzioni dal tuo file principale
from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import AmpDecisionTransformer
from TrainingLoop import evaluate

def load_target_files(source_path):
    """
    Legge i file da testare. Supporta sia un file di testo con i percorsi
    sia un pattern globale (es. cartella con file .npz).
    """
    # La cartella base sul tuo computer
    base_local_path = '../npz'
    
    if source_path.endswith('.txt'):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File {source_path} non trovato.")
            
        valid_paths = []
        with open(source_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Spezza il percorso originario di Kaggle usando '/'
                # Es: ['kaggle', 'input', 'dataset', 'reg_m-B', 'file123.npz']
                parts = line.split('/')
                
                if len(parts) >= 2:
                    folder = parts[-2]   # Prende la penultima parte (es. 'reg_m-A' o 'reg_m-B')
                    filename = parts[-1] # Prende l'ultima parte (es. 'file123.npz')
                    
                    # Ricostruisce il path locale: '../npz/reg_m-B/file123.npz'
                    local_path = os.path.join(base_local_path, folder, filename)
                    valid_paths.append(local_path)
                    
        return valid_paths
        
    else:
        files = sorted(glob.glob(source_path, recursive=True))
        if not files:
            raise FileNotFoundError(f"Nessun file trovato con il pattern: {source_path}")
        return files

def main(target_source, weights_path):
    # --- 1. Configurazione Ottimizzata ---
    # Rileva automaticamente l'hardware locale per massimizzare le prestazioni
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Accelerazione hardware per chip Apple Silicon
    else:
        device = torch.device('cpu')
        
    print(f"🚀 Inizializzazione test locale sul device: {device}")

    # --- 2. Selezione dei File ---
    # Puoi inserire il percorso a un file .txt oppure a una cartella es: '../npz/test_custom/*.npz'
    target_source1 = '../modelWeights/test_files_regmA+B.txt' 
    target_source2 = '../npz/reg_m-B/*.npz' 
    test_files = load_target_files(target_source)
    print(f"📂 Trovati {len(test_files)} file .npz da testare.")

    # --- 3. Preparazione Dataset ---
    # Per il test l'augment deve essere rigorosamente False
    print("⏳ Preprocessing dei file in RAM...")
    ds_test = PokemonVGCDataset(test_files, max_turn=49, augment=False, preload=True)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

    # --- 4. Inizializzazione Modello ---
    # Assicurati che questi parametri combacino esattamente con quelli usati nel training
    model = AmpDecisionTransformer(
        action_dim=360, 
        d_model=384, 
        n_heads=12, 
        depth=8, 
        max_turn=49
    )
    
    # --- 5. Caricamento dei Pesi ---
    print(f"🧠 Caricamento pesi da: {weights_path}")
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    
    # Estrae i pesi correttamente sia che tu gli dia il checkpoint intero (best_model) 
    # sia che tu gli dia il modello finale puro
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
   
    
    # # just for best model reg m b
    # # Carichi il checkpoint completo
    # checkpoint = torch.load(weights_path)

    # # Estrai e carichi solo i pesi del modello
    # model.load_state_dict(checkpoint['model_state_dict'])


    model.to(device)


    # --- 6. Esecuzione Valutazione ---
    criterion = nn.NLLLoss(reduction='none')
    
    # L'autocast (AMP) è supportato nativamente in modo stabile solo su CUDA
    use_amp = device.type == 'cuda' 
    
    print("⚔️ Inizio elaborazione delle predizioni...")
    val_loss, val_acc = evaluate(
        model=model, 
        dataloader=dl_test, 
        criterion=criterion, 
        device=device, 
        non_blocking=False, 
        use_amp=use_amp, 
        use_legal_mask=True,
        flip= False
    )

    # --- 7. Output Finale ---
    print("\n" + "="*30)
    print("       RISULTATI TEST")
    print("="*30)
    print(f"🎯 Accuracy pura:  {val_acc * 100:.2f}%")
    print(f"📉 Validation Loss: {val_loss:.4f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    target_source1 = '../modelWeights/test_files_regmA+B.txt' 
    target_source2 = '../npz/reg_m-B/*.npz' 
    weights_path1 = '../modelWeights/vgc_decision_transformer_regmA+B.pth' # o 'best_model.pth' se preferisci
    weights_path2 = '../modelWeights/best_model_regmB.pth', map_location='cpu'
    main(target_source2,weights_path1)



'''
    🚀 Inizializzazione test locale sul device: mps
    📂 Trovati 998 file .npz da testare.
    ⏳ Preprocessing dei file in RAM...
    🧠 Caricamento pesi da: ../modelWeights/vgc_decision_transformer_reg_m-A.pth
    ⚔️ Inizio elaborazione delle predizioni...

    ==============================
        RISULTATI TEST on regma test files
    ==============================
    🎯 Accuracy pura:  36.38%
    📉 Validation Loss: 2.0783
    ==============================

    ------
    🚀 Inizializzazione test locale sul device: mps
    📂 Trovati 13316 file .npz da testare.
    ⏳ Preprocessing dei file in RAM...
    🧠 Caricamento pesi da: ../modelWeights/vgc_decision_transformer_reg_m-A.pth
    ⚔️ Inizio elaborazione delle predizioni...

    ==============================
        RISULTATI TEST on regmb all
    ==============================
    🎯 Accuracy pura:  34.68%
    📉 Validation Loss: 2.1749
    ==============================
    ------
    🚀 Inizializzazione test locale sul device: mps
    📂 Trovati 2510 file .npz da testare.
    ⏳ Preprocessing dei file in RAM...
    🧠 Caricamento pesi da: ../modelWeights/vgc_decision_transformer_regmA+B.pth
    ⚔️ Inizio elaborazione delle predizioni...

    ==============================
        RISULTATI TEST
    ==============================
    🎯 Accuracy pura:  38.55%
    📉 Validation Loss: 1.9574
    ==============================

    Big_regMA on all RegMB
    Baseline zero-shot | val loss 2.2179 acc 0.339

'''