import os
import glob
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader # type: ignore

from PokemonVGCDataset import PokemonVGCDataset
from DecisionTransformer import AmpDecisionTransformer
from TrainingLoop import evaluate

def load_target_files(source_path):
    """
    It supports both global pattern and .txt file containing name of test files
    """
    # La cartella base sul tuo computer
    base_local_path = '../npz'
    
    if source_path.endswith('.txt'):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File {source_path} not found.")
            
        valid_paths = []
        with open(source_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Break the original Kaggle path using '/'
                # Ex: ['kaggle', 'input', 'dataset', 'reg_m-B', 'file123.npz']
                parts = line.split('/')
                
                if len(parts) >= 2:
                    folder = parts[-2]   # Takes the second to last part (e.g. 'reg_m-A' or 'reg_m-B')
                    filename = parts[-1] # Get the last part (e.g. 'file123.npz')
                    
                    # Rebuild local path: '../npz/reg_m-B/file123.npz'
                    local_path = os.path.join(base_local_path, folder, filename)
                    valid_paths.append(local_path)
                    
        return valid_paths
        
    else:
        files = sorted(glob.glob(source_path, recursive=True))
        if not files:
            raise FileNotFoundError(f"No files found with the pattern: {source_path}")
        return files

def main(target_source, weights_path):
    # --- Optimized Configuration ---
    # Automatically detects local hardware to maximize performance
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Hardware acceleration for Apple Silicon chips
    else:
        device = torch.device('cpu')
        
    print(f"Initializing local tests on the device: {device}")

    # --- File Selection ---
    # You can enter the path to a .txt file or a folder, e.g., '../npz/test_custom/*.npz'
    target_source1 = '../modelWeights/test_files_regmA+B.txt' 
    target_source2 = '../npz/reg_m-B/*.npz' 
    test_files = load_target_files(target_source)
    print(f"Founded {len(test_files)} file .npz to test.")

    # --- Dataset Preparation ---
    # For testing, the augmentation must be strictly False.
    print("Preprocessing file in RAM...")
    ds_test = PokemonVGCDataset(test_files, max_turn=49, augment=False, preload=True)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    # Make sure these parameters match exactly those used in training
    model = AmpDecisionTransformer(
        action_dim=360, 
        d_model=384, 
        n_heads=12, 
        depth=8, 
        max_turn=49
    )
    
    # --- Loading weights ---
    print(f"Loading weights from: {weights_path}")
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    
    # It extracts weights correctly whether you give it the full checkpoint (best_model)
    # or the final pure model
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.to(device)


    # --- Evaluation Execution ---
    criterion = nn.NLLLoss(reduction='none')
    
    # Autocast (AMP) is natively supported on CUDA only.
    use_amp = device.type == 'cuda' 
    
    print("Inizio elaborazione delle predizioni...")
    val_loss, val_acc = evaluate(
        model=model, 
        dataloader=dl_test, 
        criterion=criterion, 
        device=device, 
        non_blocking=False, 
        use_amp=use_amp, 
        use_legal_mask=True
    )

    # --- Final Output ---
    print("\n" + "="*30)
    print("       TEST RESULTS ")
    print("="*30)
    print(f"🎯 Accuracy :  {val_acc * 100:.2f}%")
    print(f"📉 Validation Loss: {val_loss:.4f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    target_sourceA = '../Test_files/test_files_reg_m-A.txt' 
    target_sourceB = '../Test_files/test_files_reg_m-B.txt' 
    target_sourceAB = '../Test_files/test_files_regmA+B.txt' 
    ts_zShotA = '../npz/reg_m-B/*.npz'
    ts_zShotB = '../npz/reg_m-A/*.npz'

    regmA = '../modelWeights/vgc_decision_transformer_regmA.pth' 
    regmB = '../modelWeights/vgc_decision_transformer_regmB.pth'
    regmAonB = '../modelWeights/vgc_decision_transformer_regmA_tunedOn_B.pth'
    regmAB = '../modelWeights/vgc_decision_transformer_regmA+B.pth'

    # print('target_sourceA,regmA')
    # main(target_sourceA,regmA)
    print('target_sourceB,regmB')
    main(target_sourceB,regmB)
    # print('ts_zShotA,regmA')
    # main(ts_zShotA,regmA)
    print('ts_zShotB,regmB')
    main(ts_zShotB,regmB)
    # print('target_sourceB,regmAonB')
    # main(target_sourceB,regmAonB)
    # print('target_sourceAB,regmAB')
    # main(target_sourceAB,regmAB)







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
    vgc_decision_transformer_Big_regmA_tunedOn_B.pth
    ==============================
       RISULTATI TEST - test reg_m-B
    ==============================
    🎯 Accuracy pura:  35.67%
    📉 Validation Loss: 2.0968
    ==============================
    
'''
