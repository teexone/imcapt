import hydra

import pytorch_lightning as L
import torchmetrics
import torch

from .log import ModelLogger
from ..util import to_sentence
from .encoder import Encoder
from .decoder import Decoder


class ImageCaption(L.LightningModule):
    """An image captioning module.

    A PyTorch Lightning module that produces captions for static images. 
    Provides training and validation routines, checkpoint loading/saving, 
    and a simple inference routine.   
    """
    
    def __init__(self,
                 attention_network_size, 
                 embedding_size, 
                 decoder_size, 
                 encoder_size, 
                 dropout_rate=.3,
                 beam_size=5,
                 vocabulary = None,
                 feature_map_size=14,
                 encoder_optimizer_args=None,
                 scheduler_args=None,
                 decoder_optimizer_args=None,
                 alpha_regularization=1,
                 fine_tune_encoder=False,
                 hydra_cfg=None,
                 *args, **kwargs) -> None:
        """Initializes a module with given hyperparameters

        Args:
            vocabulary (imcapt.data.Vocabulary, optional): 
                Object of the corresponding vocabulary words should be drawn from. Defaults to None.
            attention_network_size (int): 
                The size of an attention network (see ~imcapt.model.attention~)
            embedding_size (int): 
                The size of word embedding vector
            decoder_size (int): 
                The size of LSTM hidden layer in the decoder network (see ~imcapt.model.decoder) 
            encoder_size (int): 
                The size of feature dimension of the encoder network (see ~imcapt.model.encoder)
            dropout_rate (float, optional): 
                Dropout probability (uniform for all submodules). Defaults to .3.
            beam_size (int, optional): 
                The size of beam search vector. Defaults to 5.
            feature_map_size (int, optional): 
                The size of feature maps in the encoder network extracted with average pooling. Defaults to 14.
            encoder_optimizer_args (dict, optional): 
                Dict-like key-word arguments passed to ~torch.optim.Optimizer class 
                specified in the config for the encoder. Defaults to None.
            scheduler_args (dict, optional): 
                Dict-like key-word arguments passed to learning rate scheduler 
                specified in the config. Defaults to None.
            decoder_optimizer_args (dict, optional): 
                Dict-like key-word arguments passed to ~torch.optim.Optimizer class specified 
                in the config for the decoder. Defaults to None.
            alpha_regularization (int, optional): 
                Weight for additional loss intended to regularize attention layer expressiveness. Defaults to 1.
            fine_tune_encoder (bool, optional): 
                If set to True, encoder layers will be enabled to be trained in training stage. Defaults to False.
        """
        
    
        super().__init__()
        
        self.criterion = None
        self.vocabulary = None
        self._custom_logger_ = ModelLogger()
        self._encoder_optimizer_args = encoder_optimizer_args
        self._decoder_optimizer_args = decoder_optimizer_args
        self._scheduler_args = scheduler_args

        self.alpha_regularization = alpha_regularization
        self.automatic_optimization = False
        self.vocabulary = vocabulary

        self.save_hyperparameters(ignore=['vocabulary'])
        
        self.encoder = Encoder(
            feature_map_size=self.hparams.feature_map_size, 
            encoder_size=self.hparams.encoder_size,
            fine_tune=self.hparams.fine_tune_encoder,
            dropout=self.hparams.dropout_rate,
        )
        
        self.decoder = Decoder(
            embed_size=self.hparams.embedding_size,
            encoder_dim=self.hparams.encoder_size,
            attention_dim=self.hparams.attention_network_size,
            hidden_size=self.hparams.decoder_size,
            dropout=self.hparams.dropout_rate,
            vocabulary=self.vocabulary
        )

        self.criterion =  torch.nn.CrossEntropyLoss()
        self.bleu1 = torchmetrics.BLEUScore(1)
        self.bleu2 = torchmetrics.BLEUScore(2)
        self.bleu4 = torchmetrics.BLEUScore(4)

    # def setup(self, stage: str) -> None:
    #     if self.trainer is not None:
    #         self.vocabulary = self.trainer.datamodule.vocabulary
    #     return super().setup(stage)

    def train_batch(self, image: torch.Tensor, captions: torch.Tensor):
        """Training routine step

        Produces captions in two steps:
        
        1. Extracts features from images using encoder network
        2. Supplies extracted features to decoder network along with
               target captions (teacher forcing).

        Args:
            image (torch.Tensor): 
                Input images tensor of size (batch_size, image_channels, image_height, image_width)
            captions (torch.Tensor): 
                Captions tensor of size (batch_size, captions_length)

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
        features = self.encoder(image)
        return self.decoder.train_batch(features, captions)
    
    def forward(self, image):
        """Inference method

        Produces captions for a given image using
        beam search

        Args:
            image (torch.Tensor): 
                Input images tensor of size (batch_size, image_channels, image_height, image_width)

        Returns:
            (outputs, alphas): 
                list of pairs of produced captions accompanied by corresponding 
        """
        features = self.encoder(image)
        return self.decoder.forward(features)
    
    def training_step(self, batch, batch_idx):        
        """Training routine

        Performs a training routine in following steps:

        1. Runs ~ImageCaption.train_batch function with the given batch
        2. Packs both target and produced sequences with ~torch.nn.utils.rnn.pack_padded_sequence 
        3. Calculate loss function over packed sequences
        4. Applies alpha regularitzation 
        5. Performs logging and manual loss backwarding


        Args:
            batch (Tuple[Tensor, Tensor]): 
                A batch of images accompanied with target captions
            batch_idx (int): index of batch (not used)

        Returns:
            Tensor: computed loss
        """
        image, captions = batch
        preds, alphas, inds, ends = self.train_batch(image, captions)
        preds_ = torch.nn.utils.rnn.pack_padded_sequence(preds, ends, batch_first=True).data
        captions_ = torch.nn.utils.rnn.pack_padded_sequence(captions[inds][:, 1:], ends, batch_first=True).data
        loss = self.criterion(preds_, captions_)
        arl = self.alpha_regularization * ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss += arl
        
        for optimizer in self.optimizers():
            optimizer.zero_grad()
        
        self._custom_logger_.log({
            "train_sentences": to_sentence(preds, self.vocabulary),
            "train_loss": loss.item()}
        )

        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("ar_loss", arl.item(), prog_bar=True)
        self.log("train_loss_epoch", loss.item(), on_epoch=True)
        self.manual_backward(loss)

        for optimizer in self.optimizers():
            optimizer.step()
            
        return loss
    

    def validation_step(self, batch, batch_idx):
        """Validation routine 

        Performs validation routing in the following steps:
        
        1. Runs ~ImageCaption.forward method
        2. Cleans produces sentences
        3. Computes metrics and performs logging


        Args:
            batch (Tuple[Tensor, Tensor]): 
                A batch of images accompanied with target captions
            batch_idx (int): index of batch (not used)

        """
        image, captions = batch
        sentences = self.forward(image)

        clean_sentences = [self.vocabulary.clean(s) for s, _ in sentences]
        clean_sentences = [[self.vocabulary.get(x) for x in y] for y in clean_sentences]
        sentences = [[self.vocabulary.get(x) for x in y] for y, _ in sentences]
        target = to_sentence(captions, self.vocabulary)


        bleu1, bleu2, bleu4, cnt_ = 0, 0, 0, 0
        for produced, targets_ in zip(clean_sentences, target):
            bleu1 += self.bleu1(produced, targets_)
            bleu2 += self.bleu2(produced, targets_)
            bleu4 += self.bleu4(produced, targets_)
            cnt_ += 1

        bleu1 /= max(cnt_, 1)
        bleu2 /= max(cnt_, 1)
        bleu4 /= max(cnt_, 1)
        mapped = ["\nT: ".join([" ".join([self.vocabulary.get(x) for x in y]) for y in z]) for z in captions]

        self._custom_logger_.log({
            "validation_sentences": "\n".join([f"S: {' '.join(x)}\nT: {y}\n" for x, y in zip(sentences, mapped)]),
            "validation_bleu1": bleu1.item(),
            "validation_bleu2": bleu2.item(),
            "validation_bleu4": bleu4.item(),
        })
        self.log("val_bleu4", bleu4.item())
        self.log("val_bleu2", bleu2.item())
        self.log("val_bleu1", bleu1.item())


    def on_validation_end(self)-> None:
        """Saves logs once validation is finished
        """
        self._custom_logger_.save()
    
    def on_train_epoch_end(self) -> None:
        """Saves logs once epoch is finished
        """
        self._custom_logger_.save()

    def configure_optimizers(self):
        """Configures model optimizers

        The model uses two different optimizers
        for encoder and decoder networks.
        """
        optimizers = []
        
        if self._encoder_optimizer_args is not None:
            optimizers.append(
                hydra.utils.instantiate(self._encoder_optimizer_args, self.encoder.parameters()) 
            )
        
        if self._decoder_optimizer_args is not None:
             optimizers.append(
                 hydra.utils.instantiate(self._decoder_optimizer_args, self.decoder.parameters())
             )
        
        if self._scheduler_args is not None:
            schedulers = [hydra.utils.instantiate(self._scheduler_args, optimizer) for optimizer in optimizers]
        else: 
            return optimizers
        return optimizers, schedulers


        
       