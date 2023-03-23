import numpy as np
import torch 
import pytorch_lightning as L
from imcapt.data.vocabulary import Vocabulary
from imcapt.model.attention import Attention


class Decoder(L.LightningModule):
    """Decoder network

    An LSTM wrapper that generates captions
    word by word.
    """

    def __init__(self, 
                 embed_size: int, 
                 encoder_dim: int,
                 hidden_size: int,
                 attention_dim: int,
                 dropout: int,
                 vocabulary: Vocabulary):
        """LSTM network

        Args:
            embed_size (int): The size of embedding vector
            encoder_dim (int): The size of encoder size last layer
            hidden_size (int): The size of hidden layer in LSTM
            attention_dim (int): The size of attention network
            dropout (int): The probability of dropout
            vocabulary (Vocabulary): Vocabulary associated with the model
        """
        super(Decoder, self).__init__()
        # The vocabulary is necessary to produce words 
        self.vocabulary = vocabulary
        self.vocab_size = vocabulary.size()

        # Attention network
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        
        # Embedding layer
        self.embed = torch.nn.Embedding(self.vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = torch.nn.LSTMCell(
            input_size=encoder_dim + embed_size, 
            hidden_size=hidden_size,
        )

        # Beta Gate
        self.f_beta = torch.nn.Linear(encoder_dim, hidden_size)
        
        # Probability generation layer
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)

        # Dropout regularization
        self.dropout = torch.nn.Dropout(dropout)

        # For initalizing states for LSTM
        self.initial_hidden = torch.nn.Linear(embed_size, hidden_size)
        self.initial_cell = torch.nn.Linear(embed_size, hidden_size)



    def init_lstm(self, encoded_image: torch.Tensor):
        """Initializes LSTM state"""
        return (
            self.initial_hidden(encoded_image.mean(dim=1)).to(self.device), 
            self.initial_cell(encoded_image.mean(dim=1)).to(self.device)
        )
    

    
    def train_batch(self, encoded_images: torch.Tensor, captions: torch.Tensor):
        """Training routine function

        Executes a training step over a batch of images and captions. Uses teacher forcing by
        supplying a target word into the LSTM each time step.

        Args:
            encoded_images (torch.Tensor): 
                Features extracted from image as a tensor of size (batch_size, feature_map_size ** 2, encoder_size) 
            captions (torch.Tensor): 
                Captions as a tensor of size (batch_size, caption_length)

        Returns:
            predictions (Tensor):
                Word choice probabilities tensor of size (batch_size, captions_length, vocabulary_size)
            alphas (Tensor):
                Output from attention layer of size (batch_size, feature_map_size * feature_map_size)
            inds (Tensor):
                Indices permutation indicating an order the predictions were proceeded 
            decode_length (Tensor):
                Lengths of produced captions without paddings  
        """

        # Derive sizes
        batch_size = encoded_images.size(0)
        pixels = encoded_images.size(1)
        vocab_size = self.vocabulary.size()

        # Process captions in decreasing order
        # with respect to their lengths
        decode_length = captions.eq(self.vocabulary['<END>']).to(torch.int32).argmax(dim=-1) 
        inds = decode_length.argsort().flip(dims=(0,))
        decode_length = (decode_length)[inds].cpu().tolist()
        max_decode_length = max(decode_length)

        encoded_images = encoded_images[inds]
        captions = captions[inds]

        # Output values
        predictions = torch.zeros(batch_size, max_decode_length, vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max_decode_length, pixels).to(self.device)
        
        # LSTM initialization
        h_, c_ = self.init_lstm(encoded_images) # (L, batch_size, embeddings)
        
        # Target caption embeddings
        embeddings = self.embed(captions)

        for t in range(max_decode_length):
            # Take a mask of all captions with
            # length higher than current position t
        
            # Then, sum the mask to obtain a length of 
            # captions prefix to process
            
            # Suppose, we have captions lengths array as
            # C = {c_1, c_2, ..., c_n} and c_{i} >= c_{i+1}
            # Then, we compute mask M = {m_1, m_2, ..., m_n} where m_i in {0, 1} 
            
            # So, sum over M is equal to the number of captions that should 
            # be proccessed.
             
            b_t = sum([l > t for l in decode_length])
            
            # Compute attention-weighted encoding `awe` and 
            # corresponding alpha masks 
            awe_, alphas[:b_t, t, :] = self.attention(encoded_images[:b_t], h_[:b_t])

            # Apply gate
            gate_ = torch.nn.functional.sigmoid(self.f_beta(h_[:b_t]))

            # Change LSTM state
            h_, c_ = self.lstm(
                torch.cat([embeddings[:b_t, t, :], awe_ * gate_], dim=1),
                (h_[:b_t], c_[:b_t])
            )

            # Extend predictions
            predictions[:b_t, t, :] = self.linear(self.dropout(h_))

        return predictions, alphas, inds, decode_length


        
    def forward(self, inputs: torch.Tensor, maximum_length=20, k=5):
        """Batch inference method 

        Launches ~Decoder.infer for each image. See documentation
        for this method for details.
        """
        batch_size = inputs.size(0)
        outputs_ = []
        for idx in range(batch_size):
            outputs_.append(
                self.infer(inputs[idx], maximum_length, k)
            )
        return outputs_


    def infer(self, image: torch.Tensor, max_length=20, k=5):
        """Image inference method

        Args:
            image (torch.Tensor): 
                A single image as a tensor of size (feature_map_size ** 2, encoder_size).
            max_length (int, optional): 
                Max length for the caption. Defaults to 20.
            k (int, optional): 
                Beam search vector size. Defaults to 5.

        Returns:
            output (Tensor): 
                A generated caption as a tensor of size (X)
            alpha (Tensor): 
                An attention mask produced for each word as a tensor of size (X, feature_map_size ** 2)
        """
        # ======================================
        # `feature_map_size` will be referred as f
        # `beam_search` will be reffered as k
        # ====================================== 

        # To exploit beam search we will treat image 
        # as a batch of size `beam_search`.
        image = image.expand(k, *image.size()) # (k, f, encoder_size)
        vocab_size = self.vocabulary.size() 

        # Beam search vector that
        # stores a vector of k best captions
        # obtained as last time step
        k_prev_words = torch.LongTensor([[self.vocabulary.get('<START>')]] * k).to(self.device)
        # Attention masks per each step
        alphas = torch.zeros(k, 1, image.size(1)).to(self.device)
        # =====================================

        # Finished sequences, their scores and attention masks
        complete = list() 
        complete_scores = list()
        attention_masks = list()

        # Intermediate sequences
        sequences = k_prev_words # (k, 1)

        # Top k scores at each time step t
        top_k_scores = torch.zeros(k, 1).to(self.device) # (k, 1)

        # LSTM states
        h, c = self.init_lstm(image)

        # Let's go
        step = 1
        while True:

            # Embedding for the best k words obtained
            # at the last step
            embeddings = self.embed(k_prev_words) # (k, 1) -> (k, 1, embed_size)

            # Attention-weighted encoding and corresponding
            # attention mask
            awe, a = self.attention(image, h) # (k, f, encoder_size) -> { (k, f, encoder_size),  (k, f) }

            # Use a beta gate
            awe *= torch.nn.functional.sigmoid(self.f_beta(h)) 

            # Update LSTM state
            h, c = self.lstm(
                torch.cat([embeddings.squeeze(), awe], dim=1),
                (h, c)
            )

            # Computing scores
            # ==========================================
            # To make previously generated words matter
            # I will add previous scores to computed
            #
            # TODO: Weighted decay for sum to make previous
            #       results less valuable with time
            scores = self.linear(h) # (k, vocab_size)
            scores = torch.nn.functional.softmax(scores, dim=1) # (k, vocab_size)
            scores += top_k_scores.expand_as(scores)  

            # Computing top k scores and calculates indices
            # The process is the following:
            
            # We have k * vocab_size scores and retrieve
            # top k out of them. To be able to draw words
            # for each beam, the score tensor is flattened
            # to obtain a tensor of size (k * vocab_size) 
            # instead of (k, vocab_size) and then .topk
            # method is applied. Result is a tensor of size
            # (k) which values are constrained by [0, vocab_size)
            
            # Latter, to restore word indices withing vocabulary 
            # remainder operation is used. Since after flattening
            # the original indicies of words within each of k
            # arrays are located as
            # (1, 2, ..., vocab_size - 1, 1, 2, ..., vocab_size - 1, ...),
            # taking a remainder will map the indices back to original ones
        
            # Dividing values will restore the beam index

            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, largest=True, sorted=True)
            image_inds = top_k_words // vocab_size  # in [0, k)             of size k
            words_inds = top_k_words % vocab_size   # in [0, vocab_size)    of size k

            # ===========================================================#

            # Extending sequences with new generated sequences
            sequences = torch.cat(
                # (k, n) [+] (k, 1) => (k, n + 1)
                [sequences[image_inds.to(torch.int64)], words_inds.unsqueeze(1)],
                dim=1
            )

            # Saving attention mask 
            alphas = torch.cat(
                # (k, n, f) [+] (k, 1, f) => (k, n + 1, f)
                [alphas[image_inds.to(torch.int64)], a.unsqueeze(1)],
                dim=1
            )
        
            # Determining which sequences are already finished (with <END> token)
            incomplete_inds = (words_inds != self.vocabulary.get('<END>')).cpu().numpy()  # mask
            incomplete_inds = np.arange(len(words_inds.cpu()))[incomplete_inds]  # masked indices
            # set difference between all indices and incomplete
            complete_inds = np.setdiff1d(image_inds.cpu(), incomplete_inds)  
            
            assert len(incomplete_inds) + len(complete_inds) == k, (list(incomplete_inds), list(complete_inds),)

            # Saving complete indices
            if len(complete_inds) > 0:
                complete.extend(sequences[complete_inds].tolist())
                complete_scores.extend(top_k_scores[complete_inds])
                attention_masks.extend(alphas[complete_inds])
    
            k -= len(complete_inds)

            if k <= 0:
                break

            # Cut finished (completed)
            sequences = sequences[incomplete_inds]
            alphas = alphas[incomplete_inds]
            remains = image_inds[incomplete_inds].cpu()
            h = h[remains]
            c = c[remains]
            image = image[remains]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = words_inds[incomplete_inds].unsqueeze(1)

            # Do not generate longer than max lengths
            if step + 1 >= max_length:
                break
            step += 1

        if len(complete) == 0:
            complete = sequences.tolist()
            complete_scores = top_k_scores.tolist()
            attention_masks = alphas

        # If did not finish with <END> make it manually
        for sequence in complete:
            if sequence[-1] != self.vocabulary.get('<END>'):
                sequence += [self.vocabulary.get('<END')]

    
        idx = complete_scores.index(max(complete_scores))
        return complete[idx], attention_masks[idx]
