import hydra
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import torch
from imcapt.model.imcapt import ImageCaption
from imcapt.data.flickr import FlickrDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
import warnings


@hydra.main(version_base=None, config_path='../config/train', config_name='base')
def train(config: DictConfig):
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings('ignore')
    dl = hydra.utils.instantiate(config['data'])

    model = ImageCaption(
        **config['model'],
    )
    # print(len([*model.parameters()]))
    # optimizer = hydra.utils.instantiate(config['optimizer'], model.parameters())
    # scheduler = hydra.utils.instantiate(config['lr_scheduler'], optimizer)
    trainer = hydra.utils.instantiate(config['trainer'])
    # model.set_optimizer(optimizer)
    # model.set_scheduler(scheduler)
    trainer.fit(model, dl)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings('ignore')
    # train()

    dl = FlickrDataModule(captions_path="./datasets/captions/dataset_flickr8k.json",
                           folder_path="./datasets/flickr8/images",
                           h5_load="./datasets/h5/flickr8.hdf5",
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
        num_sanity_val_steps=100,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=4,
        logger=[logger],
        accelerator='gpu',
    )
    
    trainer.fit(model, dl)
