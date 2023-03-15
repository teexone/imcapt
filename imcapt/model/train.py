from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import torch
from imcapt.model.imcapt import ImageCaption
from imcapt.data.data import Flickr8DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
import warnings

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings('ignore')

    dl = Flickr8DataModule(captions_path="./datasets/captions/dataset_flickr8k.json",
                           folder_path="./datasets/flickr8/images",
                           h5_load="./datasets/h5",
                           max_caption_length=20,
                           batch_size=8)
    
    model = ImageCaption(
            attention_network_size=512,
            encoder_size=512,
            decoder_size=512,
            dropout_rate=.0,
            feature_map_size=64,
            embedding_size=512,
    )

    logger = CometLogger(
        api_key="Q5p5IPUko4uk4H294XWLtouyy",
        project_name="image-captioning",
        experiment_name="15_03_DayTrain"
    )

    trainer = Trainer(
        max_epochs=100,
        default_root_dir="./outputs", 
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=8,
        logger=[logger],
        callbacks=[StochasticWeightAveraging(swa_lrs=0.01)],
        accelerator='gpu',
    )
    
    trainer.fit(model, dl)
