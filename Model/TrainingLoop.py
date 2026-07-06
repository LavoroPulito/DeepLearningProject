#Training loop

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.optim import AdamW # type: ignore

def train_decision_transformer(model, dataloader, epochs, device, lr=1e-4):
    #Spostiamo il modello sul device (GPU o CPU)
    model.to(device) #model coincide con il nostro Decision Transformer 
    
    # Inizializziamo l'ottimizzatore
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Usiamo NLLLoss senza riduzione automatica per poter applicare la padding_mask
    criterion = nn.NLLLoss(reduction='none')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # 1. Spostiamo tutti i dizionari e i tensori sul device corretto
            state = {k: v.to(device) for k, v in batch['state'].items()}
            move = {k: v.to(device) for k, v in batch['move'].items()}
            battlefield = {k: v.to(device) for k, v in batch['battlefield'].items()}
            action = {k: v.to(device) for k, v in batch['action'].items()}
            
            reward = batch['reward'].to(device)
            turn = batch['turn'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # target_actions ha shape (batch_size, seq_length, 2) e contiene gli indici reali delle mosse
            target_actions = batch['target_actions'].to(device) 
            
            # 2. Azzeriamo i gradienti
            optimizer.zero_grad()
            
            # 3. Forward pass
            # Passiamo tutti gli argomenti previsti dal forward del DecisionTransformer
            log_probs = model(state, move, battlefield, action, reward, turn, padding_mask)
            
            # log_probs ora ha shape (batch_size, seq_length, 2, action_dim)
            # NLLLoss di PyTorch richiede che le classi siano nella seconda dimensione: (N, C, d1, d2, ...)
            # Riorganizziamo il tensore in (batch_size, action_dim, seq_length, 2)
            log_probs_transposed = log_probs.permute(0, 3, 1, 2)
            
            # 4. Calcolo della Loss
            loss = criterion(log_probs_transposed, target_actions)
            
            # loss ha shape (batch_size, seq_length, 2)
            # Applichiamo la maschera per ignorare i turni fittizi creati nel batch
            # La padding_mask ha shape (batch_size, seq_length), la espandiamo per coprire le 2 azioni
            expanded_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 2)
            
            # Moltiplichiamo la loss per la maschera (azzera la loss dei turni di padding) e facciamo la media
            masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum()
            
            # 5. Backward pass e ottimizzazione
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss Media: {avg_loss:.4f}")

    return model
