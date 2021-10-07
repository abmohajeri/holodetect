from typing import List
import torch
from torchtext.experimental.vectors import FastText
from torchtext.data.utils import get_tokenizer
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from detection.features import BaseExtractor
from utils import RowBasedValue

fasttext = FastText()
tokenizer = get_tokenizer("spacy")


class AlphaFeatureExtractor(BaseExtractor):
    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        pass

    # Attribute Level (Character Embedding & Word Embedding)
    def extract_embedding(self, data):
        char_data = stack_and_pad_tensors(
            [
                fasttext.lookup_vectors(list(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        word_data = stack_and_pad_tensors(
            [
                fasttext.lookup_vectors(tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        return char_data, word_data

    # Tuple Level (Tuple representation)
    def extract_coval_embedding(self, data: List[RowBasedValue]):
        return stack_and_pad_tensors(
            [
                fasttext.lookup_vectors(tokenizer(' '.join(list(x.row.values()))))
                if x
                else torch.zeros(1, 300)
                for x in data
            ]
        ).tensor

    def n_features(self):
        return 300


