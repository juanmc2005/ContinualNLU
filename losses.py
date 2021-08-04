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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotFillingLoss(nn.Module):
    """
    Slot Filling loss

    :param num_labels: int
        Total number of slot labels.
    :param coeff: float
        Coefficient to weigh the loss.
    :param ignore_index: int
        Index to ignore when calculating the loss.
        MUST MATCH the ignore index used to generate batches.
    """

    def __init__(
            self,
            num_labels: int,
            coeff: float,
            ignore_index: int
    ):
        super().__init__()
        self.num_labels = num_labels
        self.coeff = coeff
        self.ignore_index = ignore_index

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            **kwargs
    ) -> torch.Tensor:
        """
        Calculates the slot filling loss.
        It takes into account the attention mask
        so unused tokens are not counted.

        :param logits: Tensor
            Logits from the model. Shape (batch_size, seq_len, num_labels)
        :param labels: Tensor
            Target labels. Shape (batch_size, seq_len)
        :param attention_mask: Tensor, optional
            Attention mask given as input to the model. Shape (batch_size, seq_len).
            Defaults to None (all tokens are taken into account).
        """
        # Take attention mask into account
        # See https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
        active_logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            ignore_tensor = torch.tensor(self.ignore_index).type_as(labels)
            active_labels = torch.where(active_loss, labels, ignore_tensor)
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=self.ignore_index)
        else:
            loss = F.cross_entropy(active_logits, labels, ignore_index=self.ignore_index)
        return self.coeff * loss


class IntentDetectionLoss(nn.Module):
    """
    Intent Detection loss

    :param num_labels: int
        Total number of intent labels.
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the intent detection loss.

        :param logits: Tensor
            Logits from the model. Shape (batch_size, num_labels)
        :param labels: Tensor
            Target labels. Shape (batch_size,)
        """
        return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))