import torch

class GPT2Config:
    def __init__(self, vocab_size):
        self.embed_size = 64
        self.num_heads = 2
        self.num_layers = 4
        self.vocab_size = vocab_size
        self.block_size = 256
        self.lr = 3e-4
        self.batch_size = 32 # use a smaller batch size if you run out of memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path = 'models/model.pth'
        self.num_epochs = 10

        self.patience = 5  # Early stopping patience
        self.grad_clip = 1.0  # Gradient clipping threshold
        
        # Learning rate scheduler configs
        self.use_scheduler = True
        self.scheduler_type = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'
        self.lr_patience = 2  # For ReduceLROnPlateau
        self.lr_factor = 0.5  # For ReduceLROnPlateau
        self.step_size = 1  # For StepLR
        self.gamma = 0.1  # For StepLR

        #logging
        self.log_dir = "logs"