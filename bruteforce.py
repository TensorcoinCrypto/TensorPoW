# Created by Arav Jain on 10/26/2024
# bruteforce.py

import hash
import torch

# Tensor settings
dtype = torch.float32
device = 'cuda'

# Retrieve variables and function results from hash.py
sender_address = hash.sender_address
recipient_address = hash.recipient_address
transaction_amount = hash.transaction_amount
rand_seed = hash.rand_seed

# Unpack the tuple returned by generate_hashed_block
sender_address, recipient_address, transaction_amount, seed, transaction_tensor, hashed_block = hash.generate_hashed_block(
    sender_address, recipient_address, transaction_amount, rand_seed)

# Batch settings
total_seeds = 65536
batch_size = 512  # Adjust based on your GPU's memory capacity
threshold = 0.01  # Tolerance for comparison

# Move tensors to GPU
hashed_block = hashed_block.to(device)
transaction_tensor = transaction_tensor.to(device)

# Start batch processing
for batch_start in range(0, total_seeds, batch_size):
    batch_end = min(batch_start + batch_size, total_seeds)
    batch_seeds = list(range(batch_start, batch_end))
    actual_batch_size = batch_end - batch_start

    # Prepare tensors to hold divisor matrices
    divisor_tensors = torch.empty((actual_batch_size, 40, 40), dtype=dtype, device=device)

    # Generate divisor matrices for each seed in the batch
    for idx, seed in enumerate(batch_seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        divisor_tensors[idx] = torch.rand(size=(40, 40), generator=gen, dtype=dtype, device=device)

    # Solve the linear systems without inverting matrices
    # For each sample in the batch, we need to solve:
    # result * divisor_tensor = hashed_block
    # Transpose the equation to:
    # divisor_tensor^T * result^T = hashed_block^T

    # Transpose matrices for solving
    divisor_tensors_T = divisor_tensors.transpose(1, 2)
    hashed_block_T = hashed_block.t()

    # Expand hashed_block_T to match batch size
    hashed_block_T_expanded = hashed_block_T.unsqueeze(0).expand(actual_batch_size, -1, -1)

    try:
        # Solve the linear systems
        result_T = torch.linalg.solve(divisor_tensors_T, hashed_block_T_expanded)
    except RuntimeError:
        # Handle singular matrices using least squares
        result_T, _ = torch.linalg.lstsq(divisor_tensors_T, hashed_block_T_expanded)

    # Transpose result_T to get result
    result = result_T.transpose(1, 2)

    # Compare each result in the batch with transaction_tensor
    diff = torch.abs(result - transaction_tensor)
    max_diff_per_sample = diff.view(actual_batch_size, -1).max(dim=1).values

    # Identify indices where the max difference is within the threshold
    matching_indices = (max_diff_per_sample <= threshold).nonzero(as_tuple=True)[0]

    # If any matches are found, process them
    if matching_indices.numel() > 0:
        for idx in matching_indices:
            matching_seed = batch_seeds[idx.item()]
            print("Transaction verified!")
            print(f"Actual transaction amount: {float(transaction_amount)}")
            print(f"Actual seed: {int(seed)}")
            result_tensor = result[idx]

            # Extract calculated transaction amount from the result tensor
            calculated_transaction_amount = ""
            for j in range(20):
                calculated_transaction_amount += str(round(result_tensor[j, 39].item()))
            calculated_transaction_amount += "."
            for j in range(20):
                calculated_transaction_amount += str(round(result_tensor[j+20, 39].item()))
            print(f"Calculated transaction amount: {float(calculated_transaction_amount)}")
            print(f"Calculated seed: {matching_seed}")
        break  # Exit the loop on successful verification
