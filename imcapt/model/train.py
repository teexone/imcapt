import os
import hydra
import warnings
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from imcapt.model.imcapt import ImageCaption
from imcapt.data.vocabulary import Vocabulary
from pytorch_lightning import Trainer

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'config'), config_name='base')
def train(config: DictConfig):
    """Training launcher
    
    Function initializes all the necessary objects for training
    routine and launches it (~pytorch_lightning.trainer.Trainer.fit).

    List of objects:

    - Model. The model initialized either from checkpoint or with 
        specified arguments.
    - Optimizers: Two different optimizers for encoder and decoder submodules
    - Learning Rate Scheduler: A learning rate scheduler initialized from config
    - Loggers: All suitable loggers for PyTorch Lightning trainer
    - Trainer: PyTorch Lightning trainer object


    Args:
        config (DictConfig): Hydra configuration for each of the objects
    """
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings('ignore')
    dl = hydra.utils.instantiate(config['data'])
    
    model_args = dict(config['model'])
    if 'encoder_optimizer' in config:
        model_args['encoder_optimizer_args'] = config['encoder_optimizer']
    if 'decoder_optimizer' in config:
        model_args['decoder_optimizer_args'] = config['decoder_optimizer']
    if 'lr_scheduler' in config:
        model_args['scheduler_args'] = config['lr_scheduler']
    
    vocabulary = Vocabulary.from_h5(config['data']['h5_load'])
    model = ImageCaption(**model_args, vocabulary=vocabulary)
  
    if 'from_checkpoint' in config:
        trainer = Trainer(**config['trainer'])
        trainer.fit(model, ckpt_path=config['from_checkpoint'])
        return

    checkpoint = ModelCheckpoint(
        dirpath='./checkpoints',
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

    if 'loggers' in config:
        loggers = [hydra.utils.instantiate(logger) for logger in config['loggers']]
    else:
        loggers = []
    
    trainer = Trainer(**config['trainer'], logger=loggers, callbacks=[checkpoint])
    trainer.fit(model, dl)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings('ignore')
    train()
