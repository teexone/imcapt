import torch
from imcapt.data.vocabulary import Vocabulary

def to_sentence(data: torch.Tensor, vocab: Vocabulary):
    if len(data.size()) == 3:
        _, data = data.max(dim=2)

    sentences = []
    for i in range(data.size(0)):
        sentences.append([vocab.get(x) for x in vocab.clean(data[i])])    
    return sentences