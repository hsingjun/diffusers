import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, scaler, device, lr_scheduler=None, max_grad_norm=1.0):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            _, logits = model(x, mask=mask, return_tokens=True)
            loss = ce(logits, y)

        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        _, logits = model(x, mask=mask, return_tokens=True)
        loss = ce(logits, y)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()

    return total_loss / total, correct / total

