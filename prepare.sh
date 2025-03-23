#!/bin/bash
# Script to install all required packages for speaker diarization
# Install PyTorch (CPU or CUDA version based on availability)
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
echo "Installing PyAnnote and other dependencies..."
pip install pyannote.audio
pip install ffmpeg-python
pip install tqdm
pip install onnxruntime-gpu
pip install onnx soundfile
pip install transformers
pip install librosa

#token
# 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