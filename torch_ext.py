import torch

def matdiv(dividend, divisor):
    divisor_inv = torch.inverse(divisor)
    result = torch.matmul(dividend, divisor_inv)
    return result
