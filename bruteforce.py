# Created by Arav Jain on 10/26/2024
# bruteforce.py - simple verification script

import hashlib
import hash

# Use the same seed that produced the hashing tensor
seed = hash.rand_seed

# Regenerate the transaction tensor and hashed block
sender_address, recipient_address, transaction_amount, _, transaction_tensor, hashed_block_check, tx_id_check = hash.generate_hashed_block(
    hash.sender_address, hash.recipient_address, hash.transaction_amount, seed
)

# Compare computed hash with the stored one
if hash.to_numpy(hashed_block_check).tolist() == hash.to_numpy(hash.hashed_block).tolist():
    print("Transaction verified!")
    print(f"Actual transaction amount: {float(transaction_amount)}")
    print(f"Actual seed: {int(seed)}")
    print(f"Transaction ID: {tx_id_check}")
    print(f"Transaction computation time: {hash.transaction_time:.6f} seconds")
else:
    print("Transaction verification failed")
