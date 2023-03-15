import numpy as np
import torch 
import pytorch_lightning as L
from imcapt.data.data import Vocabulary
from imcapt.model.attention import Attention


class Decoder(L.LightningModule):
    """Decoder network translates an image to caption"""

    def __init__(self, 
                 embed_size: int, 
                 encoder_dim: int,
                 hidden_size: int,
                 attention_dim: int,
                 dropout: int,
                 vocabulary: Vocabulary):
        """LSTM network

        Args:
            embed_size:
                A size (int) of embedding vector
            hidden_size:
                A size (int) of hidden state vector
            lstm_layer_size: 
                A size of LSTM network
            vocabulary:
               An instance of Vocabulary object that stores encountered words
        """
        super(Decoder, self).__init__()
        self.vocabulary = vocabulary
        self.vocab_size = vocabulary.size()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        
        self.f_beta = torch.nn.Linear(hidden_size, encoder_dim)
        
        # Embedding layer
        self.embed = torch.nn.Embedding(self.vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = torch.nn.LSTMCell(
            input_size=encoder_dim + embed_size, 
            hidden_size=hidden_size,
        )

        self.f_beta = torch.nn.Linear(encoder_dim, hidden_size)
        
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

        # For initalizing states for LSTM
        self.initial_hidden = torch.nn.Linear(embed_size, hidden_size)
        self.initial_cell = torch.nn.Linear(embed_size, hidden_size)



    def init_lstm(self, encoded_image: torch.Tensor):
        """Initializes LSTM state"""
        return (
            # self.initial_hidden(encoded_image).unsqueeze(0).repeat(self.lstm_layer_size, 1, 1).to(self.device), 
            # self.initial_hidden(encoded_image.mean(dim=1)).to(self.device), 
            # self.initial_cell(encoded_image).unsqueeze(0).repeat(self.lstm_layer_size, 1, 1).to(self.device)
            # self.initial_cell(encoded_image.mean(dim=1)).to(self.device)
            torch.zeros_like(encoded_image.mean(dim=1)).to(self.device),
            torch.zeros_like(encoded_image.mean(dim=1)).to(self.device),
        )
    

    
    def train_batch(self, encoded_images: torch.Tensor, captions: torch.Tensor):
        """Teacher forcing technique. Used only training."""

        batch_size = encoded_images.size(0)
        pixels = encoded_images.size(1)
        vocab_size = self.vocabulary.size()
        decode_length = captions.eq(self.vocabulary['<END>']).to(torch.int32).argmax(dim=-1) - 1
        inds = decode_length.argsort().flip(dims=(0,))
        decode_length = (decode_length)[inds].cpu().tolist()
        max_decode_length = max(decode_length)

        encoded_images = encoded_images[inds]
        captions = captions[inds]

        predictions = torch.zeros(batch_size, max_decode_length, vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_decode_length, pixels).to(self.device)
        h_, c_ = self.init_lstm(encoded_images) # (L, batch_size, embeddings)
        embeddings = self.embed(captions)

        for t in range(max(decode_length)):
            b_t = sum([l > t for l in decode_length])
            awe_, alphas[:b_t, t, :] = self.attention(encoded_images[:b_t], h_[:b_t])
            gate_ = torch.nn.functional.sigmoid(self.f_beta(h_[:b_t]))
            h_, c_ = self.lstm(
                torch.cat([embeddings[:b_t, t, :], awe_ * gate_], dim=1),
                (h_[:b_t], c_[:b_t])
            )
            predictions[:b_t, t, :] = self.linear(self.dropout(h_))

        return predictions, alphas, inds, decode_length


        
    def forward(self, inputs: torch.Tensor, maximum_length=20, k=5):
        """Used only inference."""
        batch_size = inputs.shape[0]
        outputs_ = []
        for idx in range(batch_size):
            outputs_.append(
                self._forward(inputs[idx], maximum_length, k)
            )
        return outputs_

    def _forward(self, image: torch.Tensor, max_length=20, k=5):
        image = image.expand(k, *image.size())
        vocab_size = self.vocabulary.size()
        k_prev_words = torch.LongTensor([[self.vocabulary.get('<START>')]] * k).to(self.device)
        sequences = k_prev_words # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(self.device)
        complete = list()   
        complete_scores = list()
        h, c = self.init_lstm(image)
        step = 1
        while True:
            embeddings = self.embed(k_prev_words) # (k, 1, 512)
            awe, _ = self.attention(image, h)
            awe *= torch.nn.functional.sigmoid(self.f_beta(h)) 

            h, c = self.lstm(
                torch.cat([embeddings.squeeze(), awe], dim=1),
                (h, c)
            )

            scores = self.linear(h)
            scores = torch.nn.functional.softmax(scores.squeeze(), dim=1) # (k, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores

            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, largest=True, sorted=True)
            image_inds = top_k_words // vocab_size
            words_inds = top_k_words % vocab_size


            sequences = torch.cat(
                [sequences[image_inds.to(torch.int64)], words_inds.unsqueeze(1)],
                dim=1
            )
        
            incomplete_inds = (words_inds != self.vocabulary.get('<END>')).cpu().numpy()
            incomplete_inds = np.arange(len(words_inds.cpu()))[incomplete_inds]
            complete_inds = np.setdiff1d(image_inds.cpu(), incomplete_inds)

            if len(complete_inds) > 0:
                complete.extend(sequences[complete_inds].tolist())
                complete_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)
            if k == 0:
                break
            sequences = sequences[incomplete_inds]
            remains = image_inds[incomplete_inds]
            h = h[remains.cpu()]
            c = c[remains.cpu()]
            image = image[remains.cpu()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = image_inds[incomplete_inds].unsqueeze(1)

            if step > max_length:
                break
            step += 1

        if len(complete) == 0:
            complete = sequences.tolist()
            complete_scores = top_k_scores.tolist()
    
        output = complete[
            complete_scores.index(max(complete_scores))
        ]

        return output
