# Checkpoints

Place a compatible DisasterBridge checkpoint in this directory and pass its path to `inference.py` with `--checkpoint`. The loader accepts a plain PyTorch state dictionary or a dictionary containing `model_state_dict`, `state_dict`, or `model`, and rejects incompatible parameter names or tensor shapes.
