# TensorPoW
The Fastest Blockchain, Powered by GPUs

## Usage

1. Run `python bruteforce.py` and enter a transaction amount when prompted.
2. The script generates a random sender/recipient, computes a hashed block using a
   24‑bit seed and verifies the transaction immediately.
3. After verification, the script prints how long the transaction computation took.

PyTorch is used if available for GPU acceleration. If it is not installed, the
scripts fall back to CPU with `numpy`.
