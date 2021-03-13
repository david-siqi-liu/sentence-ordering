from src.utils import *
from src.config import *

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, WeightedRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from itertools import permutations
from collections import namedtuple, Counter
from math import floor
import torch
import pickle


def load_dataset(file):
    dataset = pickle.load(open(file, 'rb'))
    print("{:s} loaded. Size: {:d}".format(file, len(dataset)))
    return dataset


def load_labeled_dataset():
    return load_dataset(args['data_dir'] + args['labeled_filename'])


def load_pred_dataset():
    return load_dataset(args['data_dir'] + args['pred_filename'])


def split_labeled_set(dataset, val_size=0.2):
    set_seed()
    train_set, val_set = train_test_split(dataset, test_size=val_size)
    print("train_set size: {:d}\nval_set size: {:d}".format(
        len(train_set), len(val_set)
    ))
    return train_set, val_set


def get_fsents(dataset):
    fsents = []
    for i, doc in enumerate(dataset):
        for j, text in enumerate(doc['sentences']):
            if doc['indexes'][j] == 0:
                fsents.append(text)
    return fsents


Sentence = namedtuple('Sentence', ('guid', 'text', 'index', 'label'))


def clean_text(text):
    return text


class FirstSentenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.sentences, self.labels = self._get_sentences(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _get_sentences(self, data):
        sentences = []
        labels = []
        for i, doc in enumerate(data):
            for j, text in enumerate(doc['sentences']):
                guid = '{:d}-{:d}'.format(i, j)
                if 'indexes' in doc:
                    index = doc['indexes'][j]
                else:
                    index = -1
                label = 1 if index == 0 else 0
                labels.append(label)
                sentences.append(Sentence(guid, text, index, label))
        print("Dataset loaded. Size: {:d}".format(len(sentences)))
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        cleaned_text = clean_text(sentence.text)

        encoding = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,  # single sentence, do not need
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'guid': sentence.guid,
            'text': sentence.text,
            'cleaned_text': cleaned_text,
            'index': sentence.index,
            'label': torch.LongTensor([sentence.label]),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


SentencePair = namedtuple('SentencePair',
                          ('guid', 'text_a', 'text_b',
                           'index_a', 'index_b', 'label')
                          )


class SentencePairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.sentence_pairs, self.labels = self._get_sentence_pairs(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _get_sentence_pairs(self, data):
        sentence_pairs = []
        labels = []
        for i, doc in enumerate(data):
            # all pairwise permutations of the sentences
            perms = list(permutations(enumerate(doc['sentences']), 2))
            for (a, text_a), (b, text_b) in perms:
                guid = "{:d}-{:d}-{:d}".format(i, a, b)
                if 'indexes' in doc:
                    index_a = doc['indexes'][a]
                    index_b = doc['indexes'][b]
                    label = 1 if (index_b - index_a) == 1 else 0
                else:
                    index_a, index_b, label = -1, -1, -1
                labels.append(label)
                sentence_pairs.append(SentencePair(
                    guid, text_a, text_b, index_a, index_b, label
                ))
        print("Dataset loaded. Size: {:d}".format(len(sentence_pairs)))
        return sentence_pairs, labels

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, index):
        sentence_pair = self.sentence_pairs[index]
        cleaned_text_a = clean_text(sentence_pair.text_a)
        cleaned_text_b = clean_text(sentence_pair.text_b)

        encoding = self.tokenizer.encode_plus(
            cleaned_text_a,
            cleaned_text_b,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if len(encoding['input_ids']) > self.max_length:
            cleaned_text_a, cleaned_text_b = truncate_texts(
                cleaned_text_a,
                cleaned_text_b,
                self.tokenizer,
                self.max_length
            )
            encoding = self.tokenizer.encode_plus(
                cleaned_text_a,
                cleaned_text_b,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            assert len(encoding['input_ids']) <= self.max_length

        return {
            'guid': sentence_pair.guid,
            'text_a': sentence_pair.text_a,
            'text_b': sentence_pair.text_b,
            'cleaned_text_a': cleaned_text_a,
            'cleaned_text_b': cleaned_text_b,
            'index_a': sentence_pair.index_a,
            'index_b': sentence_pair.index_b,
            'label': torch.LongTensor([sentence_pair.label]),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def truncate_texts(text_a, text_b, tokenizer, max_length):
    # exclude [CLS] and [SEP]
    encoding_a = tokenizer.encode(text_a)
    length_a = len(encoding_a) - 2
    encoding_b = tokenizer.encode(text_b)
    length_b = len(encoding_b) - 2
    # exclude [CLS] and [SEP] x 2
    max_length_each = floor((max_length - 3) / 2)
    if length_a < max_length_each:
        # a is okay, truncate b
        text_b = tokenizer.decode(
            encoding_b[1:max_length - 3 - length_a + 1]
        )
    elif length_b < max_length_each:
        # b is okay, truncate a
        text_a = tokenizer.decode(
            encoding_a[1:max_length - 3 - length_b + 1]
        )
    else:
        # truncate both
        text_a = tokenizer.decode(
            encoding_a[1:max_length_each + 1]
        )
        text_b = tokenizer.decode(
            encoding_b[1:max_length_each + 1]
        )
    return text_a, text_b


def get_weights_for_balanced_classes(dataset, target_ratio=0.3):
    labels = dataset.labels
    label_counts = Counter(labels)
    num_ones = label_counts[1]
    num_zeros = label_counts[0]
    weight_one = target_ratio / (1 - target_ratio) * (num_zeros / num_ones)
    print("Weight for Ones: {:.4f}".format(weight_one))
    return [weight_one if label == 1 else 1 for label in labels], weight_one