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

from dataclasses import dataclass
from typing import Text, Optional, List, Tuple

from pytorch_lightning import Trainer

from dataset import ATISLanguage
from model import MultiBERTForNLU

from config import Config


@dataclass
class ContinualTrainingState:
    """
    A snapshot of the current state of continual training at a specific point in time.

    :param continual_trainer: ContinualTrainer
        The continual trainer object that is training on the language sequence
    :param model: MultiBERTForNLU
        The model object as it is at this point of training
    :param lang_sequence: List[Text]
        The sequence of training languages
    :param lang_index: Optional[int]
        The index of the current language if available
    :param lang_trainer: Optional[Trainer]
        The PyTorch Lightning trainer used for the current language if available
    """
    # Type is specified like this to avoid circular dependencies
    continual_trainer: 'ContinualTrainer'
    model: MultiBERTForNLU
    lang_sequence: List[Text]
    lang_index: Optional[int]
    lang_trainer: Optional[Trainer]


class ContinualTrainingPlugin:
    """
    Defines callbacks to dynamically add features to the continual training loop
    """

    def on_before_sequence(self, state: ContinualTrainingState):
        pass

    def on_before_language(self, state: ContinualTrainingState):
        pass

    def on_after_language(self, state: ContinualTrainingState):
        pass

    def on_after_sequence(self, state: ContinualTrainingState):
        pass


class LanguageTrainer:
    def __init__(self, config: Config):
        self.config = config.train

    def get_loader(self, language: ATISLanguage, split: str):
        return language.get_loader(
            split, self.config.batch_size, split == "train", self.config.num_workers
        )

    def train(
            self,
            model: MultiBERTForNLU,
            lang_code: str,
            pos: int,
            **kwargs
    ) -> Tuple[Trainer, MultiBERTForNLU]:
        raise NotImplementedError
