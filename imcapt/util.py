import torch
from imcapt.data.data import Vocabulary

def to_sentence(data: torch.Tensor, vocab: Vocabulary):
    if len(data.shape) == 3:
        _, data = data.max(dim=2)

    sentences = []
    for i in range(data.shape[0]):
        sentences.append([])
        for j in range(data.shape[1]):
            sentences[-1].append(vocab.get(int(data[i, j])))
    
    return sentences