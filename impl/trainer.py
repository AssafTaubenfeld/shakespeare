"""Training utilities for GPT-2 model"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from collections import deque
import os

class Trainer:
    """
    Trainer class for GPT-2 model.
    Handles training loop, validation, checkpointing, and logging.
    
    Example:
        trainer = Trainer(model, optimizer, config)
        results = trainer.train(train_loader, val_loader)
    """
    
    def __init__(self, model, optimizer, config):
        """
        Args:
            model: GPT2 model instance
            optimizer: PyTorch optimizer
            config: GPT2Config instance with training parameters
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.train_losses = deque(maxlen=1000)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.config.device), y.to(self.config.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, loss = self.model(x, y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
                
                # Track losses
                loss_item = loss.item()
                self.train_losses.append(loss_item)
                epoch_losses.append(loss_item)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_item:.4f}",
                    'avg_loss': f"{np.mean(self.train_losses):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to tensorboard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('Loss/train_step', loss_item, self.global_step)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                self.global_step += 1
        
        return np.mean(epoch_losses)
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", leave=False) as pbar:
            for x, y in pbar:
                x, y = x.to(self.config.device), y.to(self.config.device)
                logits, loss = self.model(x, y)
                val_losses.append(loss.item())
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join('models', f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join('models', 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! Val loss: {val_loss:.4f}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            
        Returns:
            dict: Training results and metrics
        """
        print("Starting training...")
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        print(f"Device: {self.config.device}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            avg_train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            avg_val_loss = self.validate(val_loader, epoch)
            
            # Log epoch metrics
            self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {self.config.patience} epochs without improvement")
                break
        
        # Cleanup
        self.writer.close()
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'epochs_trained': epoch + 1
        }