class ModelConfiguration:
    def __init__(self, 
                 attention_dim=None, 
                 embedding_dim=None, 
                 decoder_dim=None, 
                 encoder_dim=None, 
                 dropout_rate=None,
                 beam_size=3,
                 feature_map_size=14,
                 vocabulary=None
                 ) -> None:
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.dropout_rate = dropout_rate
        self.beam_size = beam_size
        self.feature_map_size = feature_map_size
        self.vocabulary = vocabulary

    