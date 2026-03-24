# polyp-segmentation

# Project Setup Guide

# Install Python
- **Windows:** Download the installer from [python.org] : version: 3.10.10

**Verify installation:**
Open your terminal and run: python --version

# Create a Virtual Environment
1. command to create: 
    python -m venv .venv

# Activate the Environment
1. command to Activate virtual Environment: 
    .venv\Scripts\activate
2. command to Deactivate virtual Environment: 
    deactivate

# Install Requirements
    pip install --upgrade pip
    pip install -r requirements.txt

# DATASET
https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip


# Training
command: python train.py

# Training Details:
    Input size: 256×256
    Optimizer: Adam
    Learning rate: 3e-5
    Loss function: Dice-based loss
    
# Augmentations:
    Horizontal/vertical flips
    Rotation
    Scaling
    Color jitter