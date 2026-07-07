#Training loop



# Su Kaggle, DEVI salvare in /kaggle/working/ affinché i file 
# vengano conservati alla fine dell'esecuzione
SAVE_DIR = '/kaggle/working/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, loss, filename="latest_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    path = os.path.join(SAVE_DIR, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato: {path}")

def load_checkpoint(filepath, model, optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint caricato! Riprendo dall'epoca {epoch} con loss {loss:.4f}")
        return epoch + 1 # Riprendi dall'epoca successiva
    else:
        print("Nessun checkpoint trovato. Inizio da zero.")
        return 0

def train_decision_transformer(model, dataloader_training, dataloader_validation, num_epochs, device, lr=1e-4):
    #Spostiamo il modello sul device (GPU o CPU)
    model.to(device) #model coincide con il nostro Decision Transformer 
    
    # Inizializziamo l'ottimizzatore
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Usiamo NLLLoss senza riduzione automatica per poter applicare la padding_mask
    criterion = nn.NLLLoss(reduction='none')

    start_epoch = load_checkpoint('/kaggle/working/checkpoints/latest_checkpoint.pth', model, optimizer)

    best_val_loss = float('inf')  # Per tenere traccia della migliore loss di validazione
    
    for epoch in range(start_epoch, num_epochs):
        #TRAINING LOOP
        model.train()
        total_training_loss = 0.0
        
        for batch in dataloader_training:
            # 1. Spostiamo tutti i dizionari e i tensori sul device corretto
            state = {k: v.to(device) for k, v in batch['state'].items()}
            move = {k: v.to(device) for k, v in batch['move'].items()}
            battlefield = {k: v.to(device) for k, v in batch['battlefield'].items()}
            action = {k: v.to(device) for k, v in batch['action'].items()}
            
            reward = batch['reward'].to(device)
            turn = batch['turn'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            # 1. target_actions è un dizionario, iteriamo per spostare i tensori sul device
            target_dict = {k: v.to(device) for k, v in batch['target_actions'].items()}            
            # target_actions ha shape (batch_size, seq_length, 2) e contiene gli indici reali delle mosse
            # 2. Convertiamo i 6 componenti nell'indice piatto (da 0 a 479).
            # Le dimensioni sono (2, 2, 2, 5, 2, 6). Calcoliamo i moltiplicatori (strides):
            # move: 1
            # mega: 6
            # slot_target: 6 * 2 = 12
            # player_target: 12 * 5 = 60
            # slot_user: 60 * 2 = 120
            # player_user: 120 * 2 = 240
            
            target_actions_flat = (
                target_dict['player_user'] * 240 +
                (target_dict['slot_user'] - 1) * 120 +  # slot_user va da 1 a 2, togliamo 1 per avere 0-1
                target_dict['player_target'] * 60 +
                target_dict['slot_target'] * 12 +
                target_dict['mega'] * 6 +
                target_dict['move']
            ) 
            
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
            loss = criterion(log_probs_transposed, target_actions_flat)
            
            # loss ha shape (batch_size, seq_length, 2)
            # Applichiamo la maschera per ignorare i turni fittizi creati nel batch
            # La padding_mask ha shape (batch_size, seq_length), la espandiamo per coprire le 2 azioni
            expanded_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 2)
            
            # Moltiplichiamo la loss per la maschera (azzera la loss dei turni di padding) e facciamo la media
            masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum()
            
            # 5. Backward pass e ottimizzazione
            masked_loss.backward()
            optimizer.step()
            
            total_training_loss += masked_loss.item()

        
        avg_training_loss = total_training_loss / len(dataloader_training)

        #VALIDATION
        model.eval() #inizio valutazione
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader_validation:
                state = {k: v.to(device) for k, v in batch['state'].items()}
                move = {k: v.to(device) for k, v in batch['move'].items()}
                battlefield = {k: v.to(device) for k, v in batch['battlefield'].items()}
                action = {k: v.to(device) for k, v in batch['action'].items()}
                reward = batch['reward'].to(device)
                turn = batch['turn'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                target_dict = {k: v.to(device) for k, v in batch['target_actions'].items()}

                target_actions_flat = (
                    target_dict['player_user'] * 240 +
                    (target_dict['slot_user'] - 1) * 120 +  # slot_user va da 1 a 2, togliamo 1 per avere 0-1
                    target_dict['player_target'] * 60 +
                    target_dict['slot_target'] * 12 +
                    target_dict['mega'] * 6 +
                    target_dict['move']
                )

                log_probs = model(state, move, battlefield, action, reward, turn, padding_mask)
                log_probs_transposed = log_probs.permute(0, 3, 1, 2)
                loss = criterion(log_probs_transposed, target_actions_flat)
                expanded_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 2)
                masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum()
                
                total_val_loss += masked_loss.item()
                # Process the batch and compute validation loss

        avg_val_loss = total_val_loss / len(dataloader_validation)

        save_checkpoint(epoch, model, optimizer, avg_training_loss, "latest_checkpoint.pth")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(epoch, model, optimizer, avg_val_loss, "best_model.pth")
            print(f"Nuovo miglior modello salvato all'epoca {epoch+1} con loss di validazione {avg_val_loss:.4f}")


    return model
