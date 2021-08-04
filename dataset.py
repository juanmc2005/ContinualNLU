# MIT License
#
# Copyright (c) 2021 Université Paris-Saclay
# Copyright (c) 2021 Laboratoire national de métrologie et d'essais (LNE)
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import multiprocessing
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from config import Config

TokenizerBuilder = Callable[[Config], PreTrainedTokenizerBase]


def read_atis_tsv(file: Path, remove_bad_alignments: bool, lower: bool = False) -> pd.DataFrame:
    """
    Read a tsv file from the MultiATIS++ dataset
    :param file: Path
        path to the file
    :param remove_bad_alignments: bool
        whether to remove samples whose utterance length don't match slot labels length
    :param lower: bool
        whether to lowercase utterances
    :return:
        DataFrame with utterances and labels
    """
    data = pd.read_csv(file, sep='\t', usecols=['utterance', 'slot_labels', 'intent'])
    data.slot_labels = data.slot_labels.map(lambda xs: xs.split(' '))
    if remove_bad_alignments:
        i_keep = [
            i for i, (utt, labels)
            in enumerate(zip(data.utterance, data.slot_labels))
            if len(utt.split(' ')) == len(labels)
        ]
        data = data.iloc[i_keep, :]
    if lower:
        data.utterance = data.utterance.map(lambda u: u.lower())
    return data


class LabelEncoding:
    """
    Assign numeric values to intent labels and slot labels.
    The assignments are always the same (given the same dataset) as the labels are sorted beforehand.
    """

    def __init__(self, root: Path):
        unique_intents, unique_slots = set(), set()
        for file in root.iterdir():
            if file.is_file() and file.name.endswith('.tsv'):
                # Don't remove bad alignments so we don't miss any label
                file_data = read_atis_tsv(file, remove_bad_alignments=False)
                unique_intents.update(file_data.intent.unique())
                slots = set()
                for sent_slots in file_data.slot_labels:
                    slots.update(np.unique(sent_slots))
                unique_slots.update(slots)
        self.intent_to_int = {intent: i for i, intent in enumerate(sorted(unique_intents))}
        self.int_to_intent = {i: intent for intent, i in self.intent_to_int.items()}
        self.slot_to_int = {slot: i for i, slot in enumerate(sorted(unique_slots))}
        self.int_to_slot = {i: slot for slot, i in self.slot_to_int.items()}

    @property
    def num_intents(self):
        return len(self.int_to_intent)

    @property
    def num_slot_labels(self):
        return len(self.int_to_slot)

    def get_intent_name(self, intent_code: int) -> str:
        return self.int_to_intent[intent_code]

    def get_intent_code(self, intent: str) -> int:
        return self.intent_to_int[intent]

    def get_slot_label_name(self, slot_label_code: int) -> str:
        return self.int_to_slot[slot_label_code]

    def get_slot_label_code(self, slot_label: str) -> int:
        return self.slot_to_int[slot_label]

    def codify_slot_labels(self, slot_labels: List[str]) -> List[int]:
        return [self.get_slot_label_code(label) for label in slot_labels]

    def decodify_slot_labels(self, slot_label_codes: List[int]) -> List[str]:
        return [self.get_slot_label_name(code) for code in slot_label_codes]


class ATISLanguageSplit(Dataset):
    """
    Represents a train|dev|test split of a language from the MultiATIS++ dataset.
    It will return items with labels already codified.
    """

    def __init__(self, label_encoding: LabelEncoding, utterances: pd.Series,
                 intents: pd.Series, slot_labels: pd.Series):
        self.label_encoding = label_encoding
        self.utterances = utterances
        self.intents = intents
        self.slot_labels = slot_labels

    def intent_distribution(self) -> pd.DataFrame:
        counts = self.intents.value_counts()
        ids = pd.Series([self.label_encoding.get_intent_code(i) for i in counts.keys()])
        return pd.DataFrame({'intent_id': ids,
                             'intent': counts.keys(),
                             'occurrences': counts.values,
                             'density': counts.values / counts.sum()})

    def slot_label_distribution(self) -> pd.DataFrame:
        all_labels = []
        for labels in self.slot_labels:
            all_labels.extend([label[2:]
                               if label.startswith('B-') or label.startswith('I-') else label
                               for label in labels])
        all_labels = pd.Series(all_labels)
        counts = all_labels.value_counts()
        return pd.DataFrame({'slot_label': counts.keys(),
                             'occurrences': counts.values,
                             'density': counts.values / counts.sum()})

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return (self.utterances.iloc[idx],
                self.label_encoding.get_intent_code(self.intents.iloc[idx]),
                self.label_encoding.codify_slot_labels(self.slot_labels.iloc[idx]))


class ATISLanguage:
    """
    Represents a language from the MultiATIS++ dataset.
    It pre-loads all splits (train, dev and test) and it can provide
    a pytorch DataLoader to load batches in the correct format for mBERT.
    """

    def __init__(
            self,
            config: Config,
            lang: Union[str, List[str]],
            label_encoding: LabelEncoding,
            tokenizer: PreTrainedTokenizerBase
    ):
        self.config = config
        self.lang = lang
        self.label_encoding = label_encoding
        self.tokenizer = tokenizer
        self.train, self.dev, self.test = self.load_data()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        root = self.config.dataset.path
        remove_ba = self.config.dataset.remove_bad_alignments
        lower = self.config.dataset.do_lowercase
        lang = self.lang
        if isinstance(self.lang, str):
            lang = self.config.train.sequence if self.lang == 'ALL' else [self.lang]
        self.train, self.dev, self.test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for lng in lang:
            self.train = self.train.append(read_atis_tsv(root / f"train_{lng}.tsv", remove_ba, lower))
            self.dev = self.dev.append(read_atis_tsv(root / f"dev_{lng}.tsv", remove_ba, lower))
            self.test = self.test.append(read_atis_tsv(root / f"test_{lng}.tsv", remove_ba, lower))
        return self.train, self.dev, self.test

    def __getitem__(self, split: str) -> ATISLanguageSplit:
        assert split in ['train', 'dev', 'test'], "Split should be either `train`, `dev` or `test`"
        data = getattr(self, split)
        return ATISLanguageSplit(self.label_encoding, data.utterance, data.intent, data.slot_labels)

    def _tokenize_and_align_labels(
            self,
            sentences: List[List[str]],
            labels: List[List[int]]
    ):
        # This function is adapted from the NER huggingface script
        # See https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
        tokenized_inputs = self.tokenizer(
            sentences,
            padding='longest',
            truncation=True,
            # We use this argument because the texts in the dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_tensors='pt'
        )
        aligned_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None.
                # We set the label to the ignored index so
                # they are not taken into account in the loss function.
                if word_idx is None:
                    label_ids.append(self.config.dataset.ignore_index)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label
                # to either the current label or the ignored index,
                # depending on the label_all_subwords flag.
                elif self.config.dataset.label_all_subwords:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(self.config.dataset.ignore_index)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        return tokenized_inputs, aligned_labels

    def _collate(self, batch: List[Tuple[str, int, List[int]]]) -> Dict:
        utterances, slot_labels, intents = [], [], []
        for u, i, sl in batch:
            utterances.append(u.split(' '))
            intents.append(i)
            slot_labels.append(sl)
        utterances, labels = self._tokenize_and_align_labels(utterances, slot_labels)
        return {
            'input_ids': utterances['input_ids'],
            'attention_mask': utterances['attention_mask'] if 'attention_mask' in utterances else None,
            'token_type_ids': utterances['token_type_ids'],
            'intents': torch.LongTensor(intents),
            'slot_labels': torch.LongTensor(labels)
        }

    def get_loader(
            self,
            split: str,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: Optional[int] = None
    ) -> DataLoader:
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2
        return DataLoader(dataset=self[split],
                          batch_size=batch_size,
                          collate_fn=self._collate,
                          shuffle=shuffle,
                          num_workers=num_workers)


class MultiATIS(Dataset):
    """
    Represents the entire MultiATIS++ dataset.
    It automatically identifies available languages and
    is able to build `ATISLanguage` instances to work with.
    This is the entry point to working with the dataset.

    Example:
        > config = Config('path/to/config.yml')
        > dataset = MultiATIS(config, MultiBERTTokenizer)
        > 'FR' in dataset  # True
        > french_loader = dataset['FR'].get_loader('train', batch_size=16, shuffle=True, num_workers=2)
        > for batch in french_loader:
        >     output = model(batch)
        >     ...
    """

    def __init__(self, config: Config, tokenizer: TokenizerBuilder):
        self.config = config
        root = self.config.dataset.path
        self.available_langs = [file.stem[-2:].upper()
                                for file in root.iterdir()
                                if file.is_file() and file.stem.startswith('train')]
        self.label_encoding = LabelEncoding(root)
        self.tokenizer = tokenizer(config)

    @property
    def num_intents(self):
        return self.label_encoding.num_intents

    @property
    def num_slot_labels(self):
        return self.label_encoding.num_slot_labels

    def __contains__(self, lang_code: str) -> bool:
        return lang_code.upper() in self.available_langs

    def __getitem__(self, lang: Union[str, List[str]]) -> ATISLanguage:
        if isinstance(lang, str):
            lang = lang.upper()
            languages = self.config.train.sequence if lang == 'ALL' else [lang]
        else:
            languages = [l.upper() for l in lang]
        for lng in languages:
            assert lng in self, f"{lng} is not available in this dataset"
        return ATISLanguage(
            self.config,
            lang,
            self.label_encoding,
            self.tokenizer
        )

    def get_valid_label_idx(self, targets: List[List[int]]) -> List[List[int]]:
        return [
            [i for i, l in label if l != self.config.dataset.ignore_index]
            for label in targets
        ]
