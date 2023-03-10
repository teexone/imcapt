import torch as torch

class Attention(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_hidden_dim, attention_dim) -> None:
        super().__init__()
        # The input is an encoded image 
        self.encoder_output_fc = torch.nn.Linear(encoder_dim, attention_dim)
        # The hidden is an input from previous step from decoder (also called
        # hidden state)
        self.decoder_hidden_fc = torch.nn.Linear(decoder_hidden_dim, attention_dim)
        # Attention layer produces 
        self.attention = torch.nn.Linear(attention_dim, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_output: torch.Tensor, decoder_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes an attended version of encoded image

        Args:
            - encoder_output:  The original encoder product. (batch_size, pixel_count, encoded_features)
            - decoder_hidden:  Decoder hidden state from previous time step (batch_size, decoder_hidden)
        
        Returns:
            - attended_encoding:    `torch.Tensor`  - Attended encoder product  (batch_size, )
            - alpha:                `torch.Tensor`  - Attention matrix
        """
        x = self.encoder_output_fc(encoder_output)  # (batch_size, pixel_count, attention_dim)
        y = self.decoder_hidden_fc(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim) 
        s = x + y  # (batch_size, pixel_count, attention_dim)
        attention = self.attention(s).squeeze(dim=2) # (batch size, pixel_count)
        alpha = self.softmax(attention).unsqueeze(2) # (batch_size, pixel_count)
        # (batch_size, pixel_count, encoded_feautres) * (batch_size, pixel_count, 1)
        attended_encoding = (encoder_output * alpha.unsqueeze(2)).sum(dim=1) 
        return attended_encoding, alpha
