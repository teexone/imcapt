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
                 config: ModelConfiguration,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._config = config
        self.criterion = None
        self.vocabulary = None
        self.logger = ModelLogger()
        
    

    def setup(self, stage: str) -> None:
        self.encoder = Encoder(
            feature_map_size=self._config.feature_map_size, 
            embedding_size=self._config.embedding_dim
        )
        
        self.decoder = Decoder(
            embed_size=self._config.embedding_dim,
            lstm_layer_size=4,
            hidden_size=self._config.decoder_dim,
            vocabulary=self.trainer.datamodule.vocabulary
        )

        self.vocabulary = self.trainer.datamodule.vocabulary
        self.criterion = MaskedCrossEntropyLoss(self.vocabulary)

    def train_batch(self, image: torch.Tensor, captions: torch.Tensor):
        features = self.encoder(image)
        preds = self.decoder.train_batch(features, captions)  # (batch_size, captiongs_length, vocab_size)
        return preds
    
    def forward(self, image):
        features = self.encoder(image)
        return self.decoder.forward(features)
    
    def training_step(self, batch, batch_idx):
        image, captions = batch
        preds = self.forward(image)
        loss = self.criterion(preds.contiguous().view(-1, self.vocabulary.size()).to(self.device), captions.to(torch.int64).view(-1).to(self.device))
        # print()
        self.logger.log_metrics({"train_sentences": to_sentence(preds, self.vocabulary),
                                 "train_loss": loss.mean().item()})
        return loss
    
    
    
    def on_validation_start(self) -> None: 
        # if self.validation_log.hasHandlers():
        #     self.validation_log.removeHandler(self.handler)
        # self.handler = logging.FileHandler(f"{self.path}/val/{str(self.trainer.current_epoch)}_{str(self.ind)}.log")
        # self.handler.setLevel(level=logging.INFO)
        # self.validation_log.addHandler(self.handler)
        # self.ind += 1
        return super().on_validation_start()
    
    def validation_step(self, batch, batch_idx):
        image, captions = batch
        preds = self.forward(image)
        preds_ = preds.repeat(captions.shape[1], 1, 1)
        captions = captions.contiguous().view(-1, captions.shape[-1])
        loss = self.criterion(preds_.flatten(end_dim=-2).to(self.device), captions.flatten().to(self.device))

        self.logger.log_metrics({
            "validation_sentences": to_sentence(preds, self.vocabulary),
            "validation_loss": loss.mean().item()
        })
        # print(" ".join(map(self.vocabulary.get, output[-1].to(torch.int64))))

    def on_validation_end(self)-> None:
        # self.step
        self.logger.save()
        # for st in self.tmp:
        #     self.validation_log.info(st)    
        # self.handler.flush()
        # self.handler.close()
        # self.tmp.clear()
    
    def on_train_epoch_end(self) -> None:
        self.logger.save()

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.parameters(), lr=1e-3, momentum=.9)
    

        
       