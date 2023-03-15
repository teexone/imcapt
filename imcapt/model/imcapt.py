import numpy as np
from imcapt.model.decoder import Decoder
from imcapt.model.encoder import Encoder
from imcapt.model.configuration import ModelConfiguration

import pytorch_lightning as L
import torchmetrics
import torch 
import logging
from imcapt.model.log import ModelLogger

from imcapt.model.metric import MaskedCrossEntropyLoss
from imcapt.util import to_sentence

class ImageCaption(L.LightningModule):
    def __init__(self,
                 attention_network_size=512, 
                 embedding_size=512, 
                 decoder_size=512, 
                 encoder_size=512, 
                 dropout_rate=.3,
                 beam_size=5,
                 feature_map_size=14,
                 vocabulary=None,
                 loss=torch.nn.CrossEntropyLoss,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.criterion = None
        self.vocabulary = None
        self.logger = ModelLogger()

        self.save_hyperparameters(ignore=['vocabulary', 'loss'])
    

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.encoder = Encoder(
                feature_map_size=self.hparams.feature_map_size, 
                encoder_size=self.hparams.encoder_size,
                dropout=self.hparams.dropout_rate,
            )
            
            self.decoder = Decoder(
                embed_size=self.hparams.embedding_size,
                encoder_dim=self.hparams.encoder_size,
                attention_dim=self.hparams.attention_network_size,
                hidden_size=self.hparams.decoder_size,
                dropout=self.hparams.dropout_rate,
                vocabulary=self.trainer.datamodule.vocabulary
            )

        self.vocabulary = self.trainer.datamodule.vocabulary
        self.criterion = self.hparams.loss if hasattr(self.hparams, 'loss') else torch.nn.CrossEntropyLoss()
        self.bleu1 = torchmetrics.BLEUScore(1)
        self.bleu2 = torchmetrics.BLEUScore(2)
        self.bleu4 = torchmetrics.BLEUScore(4)

    def train_batch(self, image: torch.Tensor, captions: torch.Tensor):
        features = self.encoder(image)
        return self.decoder.train_batch(features, captions)
    
    def forward(self, image):
        features = self.encoder(image)
        return self.decoder.forward(features)
    
    def training_step(self, batch, batch_idx):
        image, captions = batch
        preds, _, inds, ends = self.train_batch(image, captions)
        preds_ = torch.nn.utils.rnn.pack_padded_sequence(preds, ends, batch_first=True).data
        captions_ = torch.nn.utils.rnn.pack_padded_sequence(captions[inds][:, 1:], ends, batch_first=True).data
        loss = self.criterion(preds_, captions_)

        self.logger.log({
            "train_sentences": to_sentence(preds, self.vocabulary),
            "train_loss": loss.item()}
        )

        self.log("train_loss", loss.item())
        return loss
    

    def validation_step(self, batch, batch_idx):
        image, captions = batch
        sentences = self.forward(image)
        clean_sentences = [self.vocabulary.clean(s) for s in sentences]
        clean_sentences = [[self.vocabulary.get(x) for x in y] for y in clean_sentences]
        sentences = [[self.vocabulary.get(x) for x in y] for y in sentences]
        targets = to_sentence(captions, self.vocabulary)


        bleu1, bleu2, bleu4, cnt_ = 0, 0, 0, 0
        for produced, targets_ in zip(clean_sentences, targets):
            bleu1 += self.bleu1(produced, targets_)
            bleu2 += self.bleu2(produced, targets_)
            bleu4 += self.bleu4(produced, targets_)
            cnt_ += 1

        bleu1 /= max(cnt_, 1)
        bleu2 /= max(cnt_, 1)
        bleu4 /= max(cnt_, 1)

        self.logger.log({
            "validation_sentences": sentences,
            "validation_bleu1": bleu1.item(),
            "validation_bleu2": bleu2.item(),
            "validation_bleu4": bleu4.item(),
        })
        self.log("val_bleu4", bleu4.item())
        self.log("val_bleu2", bleu2.item())

    def on_validation_end(self)-> None:
        self.logger.save()
    
    def on_train_epoch_end(self) -> None:
        self.logger.save()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=5e-4, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.1, 
            patience=2,
            min_lr=1e-8, 
            threshold=.1,
            eps=1e-8,
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler,  "monitor": "train_loss"}
    

        
       