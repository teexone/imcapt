from abc import ABC, abstractmethod
import os


class Downloader(ABC):
    def __init__(self, root_dir: os.PathLike) -> None:
        os.makedirs(root_dir, exist_ok=True)

    @abstractmethod
    def get(self, url: list[str]):
        pass

    
    def ispresent(self) -> bool:
        return False
