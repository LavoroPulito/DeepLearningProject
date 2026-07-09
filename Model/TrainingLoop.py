"""Decision Transformer training loop.

Optimizations:
- Automatic AMP (mixed precision) on CUDA;
- Non-blocking transfers + pin_memory (from the DataLoader);
- Flat targets pre-calculated in the Dataset (no batch arithmetic);
- Legal action mask applied to logits (softmax only on the legal
action: the Dataset guarantees that the true action is always legal);
- Action accuracy metric to verify the model's capabilities.
"""
import os
import time
import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.optim import AdamW  # type: ignore



def _base(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(save_dir, epoch, model, optimizer, loss,
                    filename="latest_checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({'epoch': epoch,
                'model_state_dict': _base(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, path)
    return path


def load_checkpoint(filepath, model, optimizer):
    if filepath and os.path.exists(filepath):
        ckpt = torch.load(filepath, map_location='cpu')
        try:
            _base(model).load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except RuntimeError as e:
            print(f'Checkpoint incompatible with this configuration, '
                  f'starting from zero. ({str(e)[:120]}...)')
            return 0
        print(f"Checkpoint loaded: I'm picking up from the era {ckpt['epoch'] + 1} "
              f"(loss {ckpt['loss']:.4f})")
        return ckpt['epoch'] + 1
    print("No checkpoint founded. starting from zero.")
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
    """Returns (loss, n_azioni_valide, n_azioni_corrette)."""
    legal = batch['legal_action_mask'] if use_mask else None
    log_probs = model(batch['state'], batch['move'], batch['battlefield'],
                      batch['action'], batch['reward'], batch['turn'],
                      batch['padding_mask'], legal_action_mask=legal)
    # NB: flattened to 2D instead of permute(0,3,1,2): the CUDA kernel of
    # nll_loss2d requires contiguous input ("grad_input must be contiguous")
    target = batch['target_flat']
    loss_el = criterion(log_probs.reshape(-1, log_probs.size(-1)),
                        target.reshape(-1)).view_as(target)      # (B, T, 2)

    mask = batch['padding_mask'].unsqueeze(-1).expand_as(loss_el).float()
    loss = (loss_el * mask).sum() / mask.sum()

    with torch.no_grad():
        pred = log_probs.argmax(dim=-1)
        correct = ((pred == target).float() * mask).sum()
    return loss, mask.sum(), correct


def evaluate(model, dataloader, criterion, device, non_blocking, use_amp,
             use_legal_mask=True):
    model.eval()
    tl = tn = tc = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = _to_device(batch, device, non_blocking)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, n, correct = _run_batch(model, batch, criterion,
                                              use_legal_mask)
            tl += loss.item() * n.item()
            tn += n.item()
            tc += correct.item()
    return tl / max(tn, 1), tc / max(tn, 1)


def set_backbone_frozen(model, frozen):
    """Freeze embedding + all transformer blocks EXCEPT the last one.
    They are always trainable: last block and test predict_action."""   
    base = _base(model)
    for p in base.token_embedding.parameters():
        p.requires_grad = not frozen
    for blk in base.tblocks[:-1]:
        for p in blk.parameters():
            p.requires_grad = not frozen


def train_decision_transformer(model, dataloader_training,
                               dataloader_validation, num_epochs, device,
                               lr=1e-4, save_dir='checkpoints',
                               resume_from=None, use_legal_mask=True,
                               warmup_epochs=0, freeze_epochs=0,
                               eval_first=False, patience=4):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # linear warmup (if required) + cosine decay up to 5% of LR
    def lr_lambda(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    non_blocking = device.type == 'cuda'
    criterion = nn.NLLLoss(reduction='none')

    start_epoch = load_checkpoint(resume_from, model, optimizer) \
        if resume_from else 0
    for _ in range(start_epoch):
        scheduler.step()          
    best_val_loss = float('inf')
    epochs_no_improve = 0
    if eval_first:
        vl, va = evaluate(model, dataloader_validation, criterion, device,
                          non_blocking, use_amp, use_legal_mask)
        print(f'Baseline zero-shot | val loss {vl:.4f} acc {va:.3f}')

    frozen = False
    for epoch in range(start_epoch, num_epochs):
        # ---- layer freezing (for fine-tuning only) ----
        if freeze_epochs and epoch < freeze_epochs and not frozen:
            set_backbone_frozen(model, True)
            frozen = True
            print(f'Backbone freezed for the firsts {freeze_epochs} epochs'
                  f'(only last block + Head trained)')
        elif frozen and epoch >= freeze_epochs:
            set_backbone_frozen(model, False)
            frozen = False
            print('Backbone unfreezed: complete fine-tuning')

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
        val_loss, val_acc = evaluate(model, dataloader_validation, criterion,
                                     device, non_blocking, use_amp,
                                     use_legal_mask)

        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.1f}s")

        save_checkpoint(save_dir, epoch, model, optimizer, train_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(save_dir, epoch, model, optimizer, val_loss,
                            "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping: val loss stopped at {best_val_loss:.4f} '
                      f'from {patience} epochs (epoch {epoch + 1})')
                break

    return model
