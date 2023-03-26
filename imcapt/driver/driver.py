import logging
from multiprocessing import Pool
import os
import signal
import sys
import time
import torch
import torchvision
import argparse
import pathlib
from ..data.vocabulary import Vocabulary
from ..model.imcapt import ImageCaption

logger = logging.Logger('status', logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def handle_sigint(*args):
    exit()

def init_model_worker(_vocabulary: Vocabulary, checkpoint: str):
    import warnings
    warnings.filterwarnings("ignore")
    global vocabulary, model, worker_logger
    worker_logger = logging.Logger("worker", level=logging.INFO)
    worker_handler = logging.StreamHandler(sys.stdout)
    worker_handler.setFormatter(logging.Formatter("[%(process)d]\t::\t%(message)s"))
    worker_logger.addHandler(worker_handler)

    vocabulary = _vocabulary
    model = ImageCaption.load_from_checkpoint(checkpoint_path=checkpoint, vocabulary=vocabulary)
    worker_logger.info("Worker is initialized.")
    
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

def process_image(img_path: str):
    global model
    model.eval()
    try:
        image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        if len(image.size()) < 4:
            image = image.unsqueeze(0)
        if torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
            image = image.to(torch.device('cuda'))
        outputs = model.forward(image)
        return outputs[0][0]
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    logger.info("Launching...")
    parser = argparse.ArgumentParser()
    parser.add_argument('watch_folder', type=pathlib.Path, help="path to folder to be watched")
    parser.add_argument('output_folder', type=pathlib.Path, help="path to folder for outputs to be put into")
    parser.add_argument('-ckpt', required=True, type=pathlib.Path, help="checkpoint path")
    parser.add_argument('-v', required=True, type=pathlib.Path, help="vocabulary hdf file")
    parser.add_argument('-i', default=100, type=int, help="time in ms of update rate")
    args = parser.parse_args()

    vocabulary = Vocabulary.from_h5(str(args.v))
    logger.info("Vocabulary found and extracted.")
    os.makedirs(args.watch_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        waiting = True
        pool = Pool(2,initializer=init_model_worker, initargs=(vocabulary, str(args.ckpt)))
        while True:
            outputs = []
            files = []
            for image in os.listdir(args.watch_folder):
                print(os.path.join(args.watch_folder, image))
                files.append(os.path.join(args.watch_folder, image))
                outputs.append(pool.apply_async(process_image, args=(files[-1],)))

            if outputs: 
                with open(os.path.join(args.output_folder, str(int(time.time() * 100))) + '.txt', "w+") as output_file:
                    for output in outputs:       
                        output_file.write(
                            " ".join(map(vocabulary.get, output.get())) + '\n',
                        )
                
                logger.info(f"Iteration completed. {len(outputs)} files processed.")
                waiting = True
            else:
                if waiting: logger.info(f"Waiting for files...")
                waiting = False


            for file in files:
                os.remove(file)

            time.sleep(int(args.i) / 1000)

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()



