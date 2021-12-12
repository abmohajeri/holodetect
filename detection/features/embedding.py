from typing import List
import torch
from torchtext.data.utils import get_tokenizer
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from detection.features import BaseExtractor
from utils import RowBasedValue
import fasttext

# Use This if You Want Wikipedia Corpora
# import fasttext.util
# fasttext.util.download_model('en', if_exists='ignore')
# fasttext = fasttext.load_model('cc.en.300.bin')

# fasttext = fasttext.train_unsupervised('data/datasets/holodetect/hospital/raw/hospital.csv', "cbow", lr=0.5, dim=300)
# fasttext = fasttext.train_unsupervised('data/datasets/holodetect/soccer/raw/soccer.csv', "cbow", lr=0.5, dim=300)
# fasttext = fasttext.train_unsupervised('data/datasets/other/toy/raw/toy.csv', "cbow", lr=0.5, dim=300)
fasttext = fasttext.train_unsupervised('data/datasets/other/toy_edited/raw/toy.csv', "cbow", lr=0.5, dim=300)
# fasttext = fasttext.train_unsupervised('data/datasets/other/restaurant/raw/restaurant.csv', "cbow", lr=0.5, dim=300)
# fasttext = fasttext.train_unsupervised('data/datasets/other/university/raw/university.csv', "cbow", lr=0.5, dim=300)

tokenizer = get_tokenizer("spacy")


class AlphaFeatureExtractor(BaseExtractor):
    def fit(self, values: List[RowBasedValue]):
        pass

    def transform(self, values: List[RowBasedValue]):
        pass

    def lookup_vectors(self, tokens: List[str]):
        if not len(tokens):
            return torch.empty(0, 0)
        emb = []
        for token in tokens:
            emb.append(fasttext.get_word_vector(token))
        return torch.tensor(emb)

    # Attribute Level (Character Embedding & Word Embedding)
    def extract_embedding(self, data):
        char_data = stack_and_pad_tensors(
            [
                self.lookup_vectors([str_value])
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        word_data = stack_and_pad_tensors(
            [
                self.lookup_vectors(tokenizer(str_value))
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ]
        ).tensor
        return char_data, word_data

    # Tuple Level (Tuple representation)
    def extract_tuple_embedding(self, data: List[RowBasedValue]):
        return stack_and_pad_tensors(
            [
                self.lookup_vectors(tokenizer(' '.join(list(x.row.values()))))
                if x
                else torch.zeros(1, 300)
                for x in data
            ]
        ).tensor

    # Dataset Level (Tuple representation)
    def extract_neighbor_embedding(self, data: List[RowBasedValue]):
        return torch.tensor([
                [fasttext.get_nearest_neighbors(str_value, k=1)[0][0]]
                if str_value
                else torch.zeros(1, 300)
                for str_value in data
            ])

    def n_features(self):
        return 300


