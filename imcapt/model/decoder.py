import torch 
import pytorch_lightning as L
from imcapt.data.data import Vocabulary

class Decoder(L.LightningModule):
    """Decoder network translates an image to caption"""

    def __init__(self, 
                 embed_size: int, 
                 hidden_size: int,
                 lstm_layer_size: int, 
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
        
        # Embedding layer
        self.embed = torch.nn.Embedding(self.vocab_size, embed_size)
        self.lstm_layer_size = lstm_layer_size
        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=2 * embed_size, 
                                  hidden_size=hidden_size,
                                  num_layers=lstm_layer_size,
                                  batch_first=True,
                                  bidirectional=False)
        
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(0.5)

        # For initalizing states for LSTM
        self.initial_hidden = torch.nn.Linear(embed_size, hidden_size)
        self.initial_cell = torch.nn.Linear(embed_size, hidden_size)



    def init_lstm(self, encoded_image: torch.Tensor):
        """Initializes LSTM state"""
        return (
            self.initial_hidden(encoded_image).unsqueeze(0).repeat(self.lstm_layer_size, 1, 1).to(self.device), 
            self.initial_cell(encoded_image).unsqueeze(0).repeat(self.lstm_layer_size, 1, 1).to(self.device)
            # torch.zeros(self.lstm_layer_size, *encoded_image.size()).to(self.device),
            # torch.zeros(self.lstm_layer_size, *encoded_image.size()).to(self.device),
        )
    
    
    def train_batch(self, encoded_images: torch.Tensor, captions: torch.Tensor):
        """Teacher forcing technique. Used only training."""
        embeddings = self.embed(captions.to(torch.int32)) # (batch_size, captions_length, embeddings)
        hidden, cell = self.init_lstm(encoded_images) # (L, batch_size, embeddings)
        # print(encoded_images.unsqueeze(0).size())
        lstm_in = torch.cat([encoded_images.unsqueeze(1).repeat(1, embeddings.shape[1], 1), embeddings], dim=-1)
        lstm_out, _ = self.lstm(lstm_in, (hidden, cell)) # (batch_size, captions_length, hidden_size)
        return self.linear(lstm_out) # (batch_size, captiongs_length, vocab_size)


        
    def forward(self, inputs: torch.Tensor, maximum_length=20):
        """Used only inference."""
        hidden, cell = self.init_lstm(inputs)
        output = torch.zeros(inputs.shape[0], maximum_length, self.vocabulary.size())
        step = 0
        inputs = inputs.unsqueeze(1)
        _start = torch.LongTensor([self.vocabulary.get("<START>")]).to(self.device)
        X = torch.cat([inputs, self.embed(_start.repeat(inputs.shape[0], 1))], dim=-1)
        while step < maximum_length:
            lstm_out, (hidden, cell) = self.lstm(X, (hidden, cell,))
            preds = self.linear(lstm_out)
            _, idx = torch.max(preds, dim=2)
            output[:, step, :] = preds.squeeze()
            X = torch.cat([inputs, self.embed(idx)], dim=-1)
            step += 1
        return output





