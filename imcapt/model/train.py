import torch
from imcapt.model.configuration import ModelConfiguration
from imcapt.model.imcapt import ImageCaption
from imcapt.data.data import Flickr8DataModule, Vocabulary
from pytorch_lightning import Trainer
import warnings

from imcapt.model.log import ModelLogger

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    warnings.filterwarnings('ignore')
    dl = Flickr8DataModule(captions_path="./datasets/captions/dataset_flickr8k.json",
                           folder_path="./datasets/flickr8/images",
                           h5_load="./datasets/h5",
                           max_caption_length=20,
                           batch_size=40)
    model = ImageCaption(
        ModelConfiguration(
            attention_dim=512,
            encoder_dim=512,
            decoder_dim=512,
            dropout_rate=.0,
            embedding_dim=512,
        )
    )

    trainer = Trainer(max_epochs=5, 
                      num_sanity_val_steps=1,
                      val_check_interval=600, 
                      accelerator='gpu',
                      logger=False)
    trainer.fit(model, dl)
