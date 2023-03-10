import torch

from imcapt.data.data import Vocabulary

class MaskedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, vocabulary: Vocabulary, masked_words=["<UNKNOWN>", "<PADDING>"]) -> None:
        super().__init__()
        weight = torch.zeros(vocabulary.size())
        weight[ list(map(vocabulary.get, masked_words)) ] = 0
        self._metric = torch.nn.CrossEntropyLoss()


    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return super().forward(input, target)