# Created by Arav Jain on 10/26/2024
# hash.py

import torch
import random

# Tensor settings
dtype = torch.float32
device = 'cuda'

# Random hashing tensor with random manual seed
rand_seed = random.randint(0, 65535) # 2^16 - 1 = 65535, a secure 16-bit integer
torch.manual_seed(rand_seed)
hashing_tensor = torch.rand(
    size=[40, 40],
    dtype=dtype,
    device=device
)
#print(hashing_tensor)

# List of possible values for each address character
address_char_possibilities = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Create random sender address
sender_address_characters = [random.choice(address_char_possibilities) for _ in range(40)]
sender_address = ""
for character in sender_address_characters:
    sender_address = sender_address + character
#print(sender_address)

# Create random recipient address
recipient_address_characters = [random.choice(address_char_possibilities) for _ in range(40)]
recipient_address = ""
for character in recipient_address_characters:
    recipient_address = recipient_address + character
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

    # Initialize a 100x100 tensor with random values based on the seed
    torch.manual_seed(seed)
    transaction_tensor = torch.rand(
        size=[40, 40],
        dtype=dtype,
        device=device
    )

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

    return sender_address, recipient_address, transaction_amount, seed, transaction_tensor, torch.matmul(transaction_tensor, hashing_tensor)

sender_address, recipient_address, transaction_amount, seed, transaction_tensor, hashed_block = generate_hashed_block(sender_address, recipient_address, transaction_amount, rand_seed)
#print(hashed_block)
