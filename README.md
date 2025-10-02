# Shakespeare GPT-2

A minimal GPT-2 implementation for generating Shakespeare-style text, built with PyTorch.

## Overview

This project implements a small-scale GPT-2 transformer model trained on Shakespeare's works. It demonstrates the core components of modern language models including multi-head attention, transformer blocks, and autoregressive text generation.

## Features

- **GPT-2 Architecture**: Multi-layer transformer with self-attention
- **Training Pipeline**: Complete training loop with validation, early stopping, and gradient clipping
- **Text Generation**: Autoregressive sampling with temperature control
- **Monitoring**: TensorBoard logging for loss and metrics
- **Flexible Configuration**: Easy-to-modify hyperparameters

## Project Structure

shakespeare/
├── impl/
│ ├── model.py # GPT-2 model implementation
│ ├── attention.py # Multi-head attention mechanism
│ ├── blocks.py # Transformer blocks
│ ├── trainer.py # Training utilities
│ ├── dataset.py # Shakespeare dataset
│ ├── data_module.py # Data loading and preprocessing
│ ├── generation.py # Text generation utilities
│ └── tokenizer.py # Character-level tokenizer
├── config.py # Training configuration
└── models/ # Saved model checkpoints

## Requirements

- PyTorch
- NumPy
- tqdm
- tensorboard

## Training

The model uses:
- Cross-entropy loss
- Adam optimizer
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping with patience
- Gradient clipping

Training progress is logged to TensorBoard in the `logs/` directory.