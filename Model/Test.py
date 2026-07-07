def test_decision_transformer(model, dataloader_test, device):
    print("Inizio la valutazione sul Test Set...")
    model.eval() # Imposta il modello in modalità valutazione (disabilita il dropout, ecc.)
    criterion = nn.NLLLoss(reduction='none')
    
    total_test_loss = 0.0
    total_correct_predictions = 0
    total_valid_actions = 0
    
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria
        for batch in dataloader_test:
            # 1. Spostiamo i tensori sul device
            state = {k: v.to(device) for k, v in batch['state'].items()}
            move = {k: v.to(device) for k, v in batch['move'].items()}
            battlefield = {k: v.to(device) for k, v in batch['battlefield'].items()}
            action = {k: v.to(device) for k, v in batch['action'].items()}
            reward = batch['reward'].to(device)
            turn = batch['turn'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            target_dict = {k: v.to(device) for k, v in batch['target_actions'].items()}
            legal_action_mask = batch['legal_action_mask'].to(device)

            # 2. Appiattiamo il target delle azioni in un singolo indice (da 0 a 479)
            target_actions_flat = (
                target_dict['player_user'] * 240 +
                (target_dict['slot_user'] - 1) * 120 +
                target_dict['player_target'] * 60 +
                target_dict['slot_target'] * 12 +
                target_dict['mega'] * 6 +
                target_dict['move']
            )
            target_actions_flat = torch.clamp(target_actions_flat, min=0, max=479)

            # 3. Forward pass
            log_probs = model(state, move, battlefield, action, reward, turn, padding_mask, legal_action_mask)
            
            # Riorganizziamo per la NLLLoss
            log_probs_transposed = log_probs.permute(0, 3, 1, 2).contiguous()
            
            # 4. Calcolo della Loss mascherata
            loss = criterion(log_probs_transposed, target_actions_flat)
            expanded_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 2)
            masked_loss = (loss * expanded_mask).sum() / expanded_mask.sum()
            total_test_loss += masked_loss.item()
            
            # 5. Calcolo dell'Accuracy
            # Prendiamo l'indice con la probabilità logaritmica più alta (la scelta del modello)
            predictions = log_probs_transposed.argmax(dim=1) # shape: (batch_size, seq_length, 2)
            
            # Confrontiamo predizioni e target
            correct_predictions = (predictions == target_actions_flat)
            
            # Contiamo solo le predizioni nei turni validi (ignorando il padding)
            total_correct_predictions += (correct_predictions * expanded_mask).sum().item()
            total_valid_actions += expanded_mask.sum().item()

    # Statistiche finali
    avg_test_loss = total_test_loss / len(dataloader_test)
    accuracy = (total_correct_predictions / total_valid_actions) * 100 if total_valid_actions > 0 else 0.0
    
    print("========================================")
    print(f"RISULTATI SUL TEST SET:")
    print(f"Loss Media: {avg_test_loss:.4f}")
    print(f"Accuracy (Azioni Esatte): {accuracy:.2f}%")
    print("========================================")
    
    return avg_test_loss, accuracy
