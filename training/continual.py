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

from __future__ import annotations

from typing import Text, Optional, List

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter

from dataset import ATISLanguage, MultiATIS
from model import MultiBERTForNLU
from utils import as_path
from config import Config
from .base import ContinualTrainingState, ContinualTrainingPlugin, LanguageTrainer


class ContinualEvaluator(ContinualTrainingPlugin):

    def __init__(self, dataset: MultiATIS, config: Config):
        self.dataset = dataset
        self.do_intent = config.train.do_intent_detection
        self.lang_sequence: Optional[List[Text]] = None
        self.summary_writer: Optional[SummaryWriter] = None
        self.intent_dev_perf: Optional[np.ndarray] = None
        self.intent_test_perf: Optional[np.ndarray] = None
        self.slot_dev_perf: Optional[np.ndarray] = None
        self.slot_test_perf: Optional[np.ndarray] = None

    @staticmethod
    def log_slot_filling_report(
            state: ContinualTrainingState,
            report_lang: int,
            report: pd.DataFrame,
            subset: str
    ):
        log_dir = as_path(state.lang_trainer.log_dir)
        lang_name = state.lang_sequence[report_lang]
        report.to_csv(log_dir / f"{report_lang}_{lang_name}_slot_report_{subset}.csv")

    def _as_df(self, matrix: np.ndarray):
        return pd.DataFrame(
            matrix,
            index=self.lang_sequence,
            columns=self.lang_sequence
        )

    def _log(self, state: ContinualTrainingState, matrix: np.ndarray, name: Text, subset: str):
        scalars = {
            lang: matrix[i, state.lang_index]
            for i, lang in enumerate(self.lang_sequence)
        }
        self.summary_writer.add_scalars(
            f"seq_{subset}_{name}",
            scalars,
            global_step=state.lang_index
        )

    def _print_matrix(self, title: Text, matrix: np.ndarray):
        print(title)
        print(self._as_df(matrix).round(3))

    def set_perf(self, task: str, subset: str, i: int, j: int, value: float):
        getattr(self, f"{task}_{subset}_perf")[i, j] = value

    def print(self):
        self._print_matrix("\nSlot Filling F1 (DEV):", self.slot_dev_perf)
        self._print_matrix("\nSlot Filling F1 (TEST):", self.slot_test_perf)
        if self.do_intent:
            self._print_matrix("\nIntent Accuracy (DEV):", self.intent_dev_perf)
            self._print_matrix("\nIntent Accuracy (TEST):", self.intent_test_perf)
        print()

    def on_before_sequence(self, state: ContinualTrainingState):
        self.lang_sequence = state.lang_sequence
        # Write logs to the same directory as the continual trainer
        log_dir = as_path(state.continual_trainer.trainer.log_dir)
        self.summary_writer = SummaryWriter(str(log_dir / "Sequence"))
        # Initialize the performance matrices
        num_langs = len(self.lang_sequence)
        shape = (num_langs, num_langs)
        self.intent_dev_perf = np.zeros(shape)
        self.intent_test_perf = np.zeros(shape)
        self.slot_dev_perf = np.zeros(shape)
        self.slot_test_perf = np.zeros(shape)

    def evaluate(self, step: int, dataset: ATISLanguage, state: ContinualTrainingState, subset: str):
        batch_size = state.continual_trainer.config.batch_size
        num_workers = state.continual_trainer.config.num_workers
        loader = dataset.get_loader(subset, batch_size, False, num_workers)
        # FIXME this way of obtaining results is deprecated
        results = state.lang_trainer.test(state.model, loader, verbose=False)[0]
        slot_f1 = results["slot_filling_f1"]
        slot_report = results["slot_filling_report"]
        # Put results in the corresponding cells of the performance matrices
        self.set_perf("slot", subset, step, state.lang_index, slot_f1)
        self.log_slot_filling_report(state, step, slot_report, subset)
        if self.do_intent:
            self.set_perf("intent", subset, step, state.lang_index, results["intent_accuracy"])

    def on_after_language(self, state: ContinualTrainingState):
        # Calculate the performance on each language of the sequence
        for i, lang in enumerate(self.lang_sequence):
            lang_dataset = self.dataset[lang]
            self.evaluate(i, lang_dataset, state, "dev")
            self.evaluate(i, lang_dataset, state, "test")
        # Do not log test performance
        self._log(state, self.slot_dev_perf, "slot_f1", "dev")
        if self.do_intent:
            self._log(state, self.intent_dev_perf, "intent_acc", "dev")
        self.print()

    def _dump_performance(self, performance: np.ndarray, filename: Text):
        path = as_path(self.summary_writer.log_dir) / filename
        self._as_df(performance).to_csv(path)

    def on_after_sequence(self, state: ContinualTrainingState):
        self._dump_performance(self.slot_dev_perf, "slot_performance_dev.csv")
        self._dump_performance(self.slot_test_perf, "slot_performance_test.csv")
        if self.do_intent:
            self._dump_performance(self.intent_dev_perf, "intent_performance_dev.csv")
            self._dump_performance(self.intent_test_perf, "intent_performance_test.csv")


class ContinualTrainer:

    def __init__(
            self,
            trainer: LanguageTrainer,
            eval_dataset: MultiATIS,
            config: Config,
            plugins: List[ContinualTrainingPlugin] = ()
    ):
        self.trainer = trainer
        self.config = config.train
        # noinspection PyTypeChecker
        self.plugins = [ContinualEvaluator(eval_dataset, config)] + list(plugins)

    def _call_plugins(
            self,
            callback_name: Text,
            model: MultiBERTForNLU,
            lang_sequence: List[Text],
            lang_index: Optional[int] = None,
            lang_trainer: Optional[Trainer] = None
    ):
        """
        Calls a specific method of all available plugins, passing the current state
        """
        state = ContinualTrainingState(self, model, lang_sequence, lang_index, lang_trainer)
        for plugin in self.plugins:
            getattr(plugin, callback_name)(state)

    def train(self, model: MultiBERTForNLU, lang_sequence: Optional[List[Text]] = None, **kwargs):
        # Resolve the training sequence. Priority: argument -> config -> random sequence
        if lang_sequence is None:
            if self.config.languages is None or not self.config.languages:
                raise ValueError("A language sequence must be provided as argument or in config.yml")
            else:
                lang_sequence = self.config.languages

        print("Training Sequence:", " -> ".join(lang_sequence))

        self._call_plugins("on_before_sequence", model, lang_sequence)

        # Iterate over the sequence of languages
        num_langs = len(lang_sequence)
        for i, lang in enumerate(lang_sequence):

            self._call_plugins("on_before_language", model, lang_sequence, i)

            print(f"Training on {lang.upper()} ({i + 1}/{num_langs})")

            # Get corresponding training and validation data
            pl_trainer, model = self.trainer.train(model, lang, i, **kwargs)

            self._call_plugins("on_after_language", model, lang_sequence, i, pl_trainer)

        self._call_plugins("on_after_sequence", model, lang_sequence)
