"""Utility functions for training and evaluation"""
import torch

@torch.no_grad()
def estimate_loss(model, eval_iters, data_loaders):
    """
    Estimate loss on multiple data splits.
    
    Args:
        model: Model to evaluate
        eval_iters: Number of iterations to average over
        data_loaders: Dict of {split_name: dataloader}
        
    Returns:
        dict: Average loss for each split
    """
    model.eval()
    out = {}
    
    for split, dl in data_loaders.items():
        losses = []
        for i, (x, y) in enumerate(dl):
            if i >= eval_iters:
                break
            x, y = x.to(model.config.device), y.to(model.config.device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0
    
    model.train()
    return out

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params
    }

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer from checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Checkpoint metadata (epoch, losses, etc.)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'global_step': checkpoint.get('global_step', 0)
    }