# pyannote3.1 + whisper_tiny
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_tiny_pyannote31.txt -w tiny

# pyannote3.1 + whisper_base
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_base_pyannote31.txt -w base

# pyannote3.1 + whisper_small
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_small_pyannote31.txt -w small

# pyannote3.0 + whisper_tiny
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_tiny_pyannote30.txt -w tiny -p pyannote/speaker-diarization-3.0

# pyannote3.0 + whisper_base
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_base_pyannote30.txt -w base -p pyannote/speaker-diarization-3.0

# pyannote3.0 + whisper_small
python run_diarization.py EastEnders_Episode_5016.ts -o results/whisper_small_pyannote30.txt -w small -p pyannote/speaker-diarization-3.0