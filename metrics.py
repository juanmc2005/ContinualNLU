import uuid
from typing import Dict, List, Text, Union

import pandas as pd
import torch
from datasets import load_metric
from pytorch_lightning.metrics import Metric
from seqeval.metrics import classification_report

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

from dataset import LabelEncoding

UNIQUE_RUN_ID = str(uuid.uuid4())


def cat_labels(old: List[List[Text]], new: List[List[Text]]) -> List[List[Text]]:
    """
    Custom concatenation of lists to keep the
    state of the metric as lists of lists.
    """
    old.extend(new)
    return old


class SlotF1(Metric):
    """
    A PyTorch Lightning metric to calculate slot filling F1 score using the seqeval script.
    The seqeval script is used via the Huggingface metrics interface.
    """

    def __init__(
            self,
            label_encoding: LabelEncoding,
            ignore_index: int,
            dist_sync_on_step=False,
            name_or_path: str = 'seqeval',
            compute_report: bool = False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.encoding = label_encoding
        self.ignore_index = ignore_index
        self.seqeval = load_metric(name_or_path, experiment_id=UNIQUE_RUN_ID)
        self.compute_report = compute_report
        self.add_state("predictions", default=[], dist_reduce_fx=cat_labels)
        self.add_state("targets", default=[], dist_reduce_fx=cat_labels)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update internal state with a new batch of predictions and targets.
        This function is called automatically by PyTorch Lightning.

        :param predictions: Tensor, shape (batch_size, seq_len, num_slot_labels)
            Model predictions per token as (log) softmax scores.
        :param targets: Tensor, shape (batch_size, seq_len)
            Slot filling ground truth per token encoded as integers.
        """
        # Get hard predictions
        predictions = torch.argmax(predictions, dim=-1)
        # Transform to list since it needs to deal with different sequence lengths
        predictions = predictions.tolist()
        targets = targets.tolist()
        # Remove ignored predictions (special tokens and possibly subtokens)
        true_predictions = [
            [self.encoding.get_slot_label_name(p) for (p, l) in zip(pred, label) if l != self.ignore_index]
            for pred, label in zip(predictions, targets)
        ]
        true_targets = [
            [self.encoding.get_slot_label_name(l) for (p, l) in zip(pred, label) if l != self.ignore_index]
            for pred, label in zip(predictions, targets)
        ]
        # Add predictions and labels to current state
        self.predictions += true_predictions
        self.targets += true_targets

    def compute(self) -> Union[torch.Tensor, Dict]:
        """
        Compute the Slot F1 score using the current state.
        """
        results = self.seqeval.compute(predictions=self.predictions, references=self.targets)
        # overall_precision, overall_recall and overall_accuracy are also available
        f1 = torch.tensor(results["overall_f1"])
        if self.compute_report:
            report = classification_report(
                y_true=self.targets, y_pred=self.predictions, output_dict=True
            )
            return {"f1": f1, "report": pd.DataFrame(report).transpose()}
        else:
            return f1
