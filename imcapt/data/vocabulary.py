import json
import h5py
from pyparsing import Iterable


class Vocabulary:   
    """Stores vocabular and operates with H5 files"""

    @staticmethod
    def from_h5(file: str|h5py.File, 
                group_path="vocabular",
                words_dict_path="words",
                inverted_dict_path="inverted"):
        """Load vocabulary from an h5 file

        Args:
            - file: 
                path to file or h5py.File itself
            - group_path: 
                path to group (str) in h5 hierarchy
            - words_dict_path: 
                path to words index dataset (str) in h5 group hierarchy
            - inverted_dict_path: 
                path to inverted words index dataset (str) in h5 group hierarchy
        
        Returns:
            Constructed Vocabulary object
        """
        if isinstance(file, str):
            file = h5py.File(file) 

        vocabulary = Vocabulary()
        vocabulary._word_map = json.loads(
            file.get(f"{group_path}/{words_dict_path}")[0]
        )
        vocabulary._inverted_word_map = json.loads(
            file.get(f"{group_path}/{inverted_dict_path}")[0]
        )
        vocabulary._words_count = len(vocabulary._word_map)
        return vocabulary

    def to_h5(  self,
                file: str|h5py.File, 
                group_path="vocabular",
                words_dict_path="words",
                inverted_dict_path="inverted"):
        """Saves vocabulary in h5 file

        Args:
            - file: 
                path to file or h5py.File itself
            - group_path: 
                path to group (str) in h5 hierarchy
            - words_dict_path: 
                path to words index dataset (str) in h5 group hierarchy
            - inverted_dict_path: 
                path to inverted words index dataset (str) in h5 group hierarchy
        """
        
        if isinstance(file, str):
            file = h5py.File(file) 
   
        group = file.require_group(group_path)
        w_map_ds = group.require_dataset(words_dict_path, shape=(1,), dtype=h5py.string_dtype())
        w_map_ds[0] = json.dumps(self._word_map)
        iw_map_ds = group.require_dataset(inverted_dict_path, shape=(1,), dtype=h5py.string_dtype())
        iw_map_ds[0] = json.dumps(self._inverted_word_map)
   
    def _next_word_id(self):
        """Generates a numeric representation for new word"""
        return self._words_count
    
    def add(self, word: str):
        if word not in self._word_map:
            id = self._next_word_id()
            self._word_map[word] = id
            self._inverted_word_map[id] = word
            self._words_count += 1

    def update(self, words: Iterable[str]):
        for word in words:
            self.add(word)

    def get(self, reference: str|int):
        if isinstance(reference, str):
            return self._word_map.get(str(reference), self._word_map['<UNKNOWN>'])
        else:
            if int(reference) in self._inverted_word_map:
                return self._inverted_word_map.get(int(reference), '<UNKNOWN>')
            if str(int(reference)) in self._inverted_word_map:
                return self._inverted_word_map.get(str(int(reference)), '<UNKNOWN>')
            return '<UNKNOWN>'

    def _init_special_tokens(self):
        for x in ["<PADDING>", "<START>", "<END>", "<UNKNOWN>",]:
            self.add(x)

    def __getitem__(self, key):
        return self.get(key)
    
    
    def __iter__(self):
        return self._word_map.__iter__()
    
    def size(self):
        return self._words_count

    def __init__(self) -> None:
        self._words_count = 0
        self._word_map = {}
        self._inverted_word_map = {}
        self._init_special_tokens()

    def clean(self, sentence: Iterable[int]):
        cleaned = []
        not_in = [
            self.get("<START>"),
            self.get("<END>"),
            self.get("<UNKNOWN>"),
            self.get("<PADDING>")
        ]
        for word in sentence:
            word = int(word)
            if word not in not_in:
                cleaned.append(word)
        return cleaned
            