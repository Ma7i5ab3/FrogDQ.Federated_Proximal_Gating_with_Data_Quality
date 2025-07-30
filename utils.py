import torch

def get_top_k_indexes(weights: torch.Tensor, k: int, largest: bool = True):
    """
        Return the top k indexes with the largest absolute value from a torch.Tensor 
    """
    abs_weights = torch.abs(weights) #Compute the absolute values
    topk = torch.topk(abs_weights, k, largest=largest, sorted=True)
    return topk.indices.tolist()