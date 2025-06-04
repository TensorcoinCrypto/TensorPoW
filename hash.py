# Created by Arav Jain on 10/26/2024
# hash.py

import hashlib
import secrets
import time

try:
    import torch
    TORCH_AVAILABLE = True
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:  # Fallback to numpy if torch isn't installed
    TORCH_AVAILABLE = False
    import numpy as np
    dtype = np.float32
    device = 'cpu'

def rand_matrix(seed):
    """Generate a 40x40 matrix based on the given seed."""
    if TORCH_AVAILABLE:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return torch.rand((40, 40), generator=gen, dtype=dtype, device=device)
    else:
        np.random.seed(int(seed) % (2 ** 32))
        return np.random.rand(40, 40).astype(dtype)

def matmul(a, b):
    return torch.matmul(a, b) if TORCH_AVAILABLE else a @ b

def to_numpy(t):
    return t.cpu().numpy() if TORCH_AVAILABLE else t

# Security: use a larger search space for the PoW seed
SEED_BITS = 24
rand_seed = secrets.randbits(SEED_BITS)
hashing_tensor = rand_matrix(rand_seed)

# List of possible values for each address character
address_char_possibilities = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Create random sender address
sender_address = "".join(secrets.choice(address_char_possibilities) for _ in range(40))
#print(sender_address)

# Create random recipient address
recipient_address = "".join(secrets.choice(address_char_possibilities) for _ in range(40))
#print(recipient_address)

#transaction_int_amount = str(random.randint(0, 99999999999999999999999999999999999999999999999999))
#transaction_dec_amount = str(random.randint(0, 99999999999999999999999999999999999999999999999999))
#transaction_amount = f"{transaction_int_amount}.{transaction_dec_amount}"
transaction_amount = str(input("Enter transaction amount in float format: "))
#print(transaction_amount)

def generate_hashed_block(sender_address, recipient_address, transaction_amount, seed):
    # Convert the wallet addresses to ASCII values
    sender_ascii_values = [ord(char) for char in sender_address]
    recipient_ascii_values = [ord(char) for char in recipient_address]

    # Initialize a 40x40 tensor with random values based on the seed
    transaction_tensor = rand_matrix(seed)

    # Fill the first [40, 1] segment with the sender's ASCII values
    for i in range(min(len(sender_ascii_values), 40)):
        transaction_tensor[i, 0] = sender_ascii_values[i]

    # Fill the first [40, 1] segment with the recipient's ASCII values
    for i in range(min(len(recipient_ascii_values), 40)):
        transaction_tensor[i, 1] = recipient_ascii_values[i]

    transaction_int_amount, transaction_dec_amount = transaction_amount.split(".")

    leading_zeros_amount = 20 - len(transaction_int_amount)
    transaction_int_amount_with_zeros = "0" * leading_zeros_amount + transaction_int_amount
    for i in range(20):
        transaction_tensor[i, 39] = float(transaction_int_amount_with_zeros[i])

    trailing_zeros_amount = 20 - len(transaction_dec_amount)
    transaction_dec_amount_with_zeros = transaction_dec_amount + "0" * trailing_zeros_amount
    for i in range(20):
        transaction_tensor[i+20, 39] = float(transaction_dec_amount_with_zeros[i])

    hashed_block = matmul(transaction_tensor, hashing_tensor)
    tx_id = hashlib.blake2b(to_numpy(hashed_block).tobytes()).hexdigest()
    return sender_address, recipient_address, transaction_amount, seed, transaction_tensor, hashed_block, tx_id

# Measure the time required to generate the hashed block
_start = time.perf_counter()
sender_address, recipient_address, transaction_amount, seed, transaction_tensor, hashed_block, transaction_id = generate_hashed_block(
    sender_address, recipient_address, transaction_amount, rand_seed
)
transaction_time = time.perf_counter() - _start
#print(hashed_block)

