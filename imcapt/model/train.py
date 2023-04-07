import os
import hydra
import warnings
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from ..data.download.download import Downloader
from .imcapt import ImageCaption
from ..data.vocabulary import Vocabulary
from ..data.util import extract_vocabulary_from_karpathy_json
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
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings('ignore')

    backbone = hydra.utils.instantiate(config['model']['backbone']['model'])
    transforms = hydra.utils.instantiate(config['model']['backbone']['transforms'])


    downloader: Downloader = hydra.utils.instantiate(config['data']['download']['downloader'])
    if not downloader.ispresent():
        downloader.get(
            config['data']['download']['get']
        )
    
    dl = hydra.utils.instantiate(config['data']['module'])


    model_args = dict(config['model'])
    del model_args['backbone']
    if 'encoder_optimizer' in config:
        model_args['encoder_optimizer_args'] = config['encoder_optimizer']
    if 'decoder_optimizer' in config:
        model_args['decoder_optimizer_args'] = config['decoder_optimizer']
    if 'lr_scheduler' in config:
        model_args['scheduler_args'] = config['lr_scheduler']

    if os.path.exists(config['data']['module']['h5_load']):
        vocabulary = Vocabulary.from_h5(config['data']['module']['h5_load'])
    else:
        vocabulary = extract_vocabulary_from_karpathy_json(config['data']['module']['captions_path'])

    if 'from_checkpoint' in config:
        model = ImageCaption.load_from_checkpoint(
            checkpoint_path=config['from_checkpoint'], 
            vocabulary=vocabulary,
        )
    else:
        model = ImageCaption(**model_args, 
                             vocabulary=vocabulary, 
                             backbone=backbone,
                             transforms=transforms)
  
    
    checkpoint = ModelCheckpoint(
        **config['checkpoint']        
    )

    if 'loggers' in config:
        loggers = [hydra.utils.instantiate(logger) for logger in config['loggers']]
    else:
        loggers = []
    
    trainer = Trainer(**config['trainer'], logger=loggers, callbacks=[checkpoint])
    if 'from_checkpoint' in config:
        trainer.fit(model, dl, ckpt_path=config['from_checkpoint'])
    else:
        trainer.fit(model, dl)

if __name__ == "__main__":

    train()
