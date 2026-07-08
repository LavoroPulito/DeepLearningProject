"""Training loop del Decision Transformer.

Ottimizzazioni:
  - AMP (mixed precision) automatica su CUDA;
  - trasferimenti non_blocking + pin_memory (dal DataLoader);
  - target flat precalcolati nel Dataset (niente aritmetica per batch);
  - maschera delle azioni legali applicata ai logits (softmax solo sul
    legale: il Dataset garantisce che l'azione vera sia sempre legale);
  - metrica di accuracy sulle azioni per verificare le capacita' del modello.
"""
import os
import time

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim import AdamW  # type: ignore


def save_checkpoint(save_dir, epoch, model, optimizer, loss,
                    filename="latest_checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, path)
    return path


def load_checkpoint(filepath, model, optimizer):
    if filepath and os.path.exists(filepath):
        ckpt = torch.load(filepath, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Checkpoint caricato: riprendo dall'epoca {ckpt['epoch'] + 1} "
              f"(loss {ckpt['loss']:.4f})")
        return ckpt['epoch'] + 1
    print("Nessun checkpoint trovato. Inizio da zero.")
    return 0


def _to_device(batch, device, non_blocking):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=non_blocking)
                      for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=non_blocking)
    return out


def _run_batch(model, batch, criterion, use_mask=True):
    """Ritorna (loss, n_azioni_valide, n_azioni_corrette)."""
    legal = batch['legal_action_mask'] if use_mask else None
    log_probs = model(batch['state'], batch['move'], batch['battlefield'],
                      batch['action'], batch['reward'], batch['turn'],
                      batch['padding_mask'], legal_action_mask=legal)
    # log_probs: (B, T, 2, action_dim); target: (B, T, 2)
    # NB: si appiattisce a 2D invece di permute(0,3,1,2): il kernel CUDA di
    # nll_loss2d richiede input contiguo ("grad_input must be contiguous")
    target = batch['target_flat']
    loss_el = criterion(log_probs.reshape(-1, log_probs.size(-1)),
                        target.reshape(-1)).view_as(target)      # (B, T, 2)

    mask = batch['padding_mask'].unsqueeze(-1).expand_as(loss_el).float()
    loss = (loss_el * mask).sum() / mask.sum()

    with torch.no_grad():
        pred = log_probs.argmax(dim=-1)
        correct = ((pred == target).float() * mask).sum()
    return loss, mask.sum(), correct


def train_decision_transformer(model, dataloader_training,
                               dataloader_validation, num_epochs, device,
                               lr=1e-4, save_dir='checkpoints',
                               resume_from=None, use_legal_mask=True):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    # cosine decay: vicino al plateau un LR piu' basso recupera ancora qualche punto
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.05)
    criterion = nn.NLLLoss(reduction='none')

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    non_blocking = device.type == 'cuda'

    start_epoch = load_checkpoint(resume_from, model, optimizer) \
        if resume_from else 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # ---- training ----
        model.train()
        t0 = time.time()
        tot_loss = tot_n = tot_correct = 0.0
        for batch in dataloader_training:
            batch = _to_device(batch, device, non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, n, correct = _run_batch(model, batch, criterion,
                                              use_legal_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item() * n.item()
            tot_n += n.item()
            tot_correct += correct.item()
        train_loss = tot_loss / tot_n
        train_acc = tot_correct / tot_n

        # ---- validation ----
        model.eval()
        vtot_loss = vtot_n = vtot_correct = 0.0
        with torch.no_grad():
            for batch in dataloader_validation:
                batch = _to_device(batch, device, non_blocking)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, n, correct = _run_batch(model, batch, criterion,
                                                  use_legal_mask)
                vtot_loss += loss.item() * n.item()
                vtot_n += n.item()
                vtot_correct += correct.item()
        val_loss = vtot_loss / max(vtot_n, 1)
        val_acc = vtot_correct / max(vtot_n, 1)

        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.1f}s")

        save_checkpoint(save_dir, epoch, model, optimizer, train_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_dir, epoch, model, optimizer, val_loss,
                            "best_model.pth")

    return model
