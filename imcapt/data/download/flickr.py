import os
from .download import Downloader


class FlickrDownloader(Downloader):
    def __init__(self, root_dir: os.PathLike, flickr_version: str) -> None:
        super().__init__(root_dir)
        self._dir = root_dir
        self._flickr = flickr_version
    
    def get(self, url: list[str]):
        import wget
        import zipfile

        assert len(url) == 2, 'requires two links'
        data_url, captions_url = url


        data_file_path = os.path.join(
            self._dir,
            "data.zip"
        )

        captions_file_path = os.path.join(
            self._dir,
            "dataset_flickr8k.json"
        )

        if os.path.exists(data_file_path):
            os.remove(data_file_path)

        if os.path.exists(captions_file_path):
            os.remove(captions_file_path)

        wget.download(data_url, data_file_path)
        with zipfile.ZipFile(data_file_path) as zf:
            zf.extractall(self._dir)

        wget.download(captions_url, captions_file_path)

    def ispresent(self) -> bool:
        directory = os.listdir(self._dir)
        if 'Images' not in directory:
            return False
        
        if self._flickr == 'flickr8':
            return "dataset_flickr8k.json" in directory
        if self._flickr == 'flickr30': 
            return "dataset_flickr30.json" in directory
        return False