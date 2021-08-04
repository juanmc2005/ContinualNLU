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
from typing import Tuple, Union, Dict, Text, List, Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import Accuracy
from transformers import BertModel, PreTrainedTokenizerBase, BertTokenizerFast

from dataset import MultiATIS
from losses import IntentDetectionLoss, SlotFillingLoss
from metrics import SlotF1
from config import Config


def MultiBERTTokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """
    This is a wrapper of the tokenizer from Huggingface
    so there's no possibility of choosing the wrong one.
    """
    return BertTokenizerFast.from_pretrained(cfg.model.name_or_path)


@dataclass
class ModelOutput:
    """
    Contains useful tensors coming from the model
    """
    word_embeddings: torch.Tensor  # (batch, seq_len, hidden)
    sentence_embeddings: torch.Tensor  # (batch, hidden)
    word_outputs: torch.Tensor  # (batch, seq_len, num_slot_labels)
    sentence_outputs: Optional[torch.Tensor]  # (batch, num_intents)

    def get_predictions(self, log: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        activation = torch.log_softmax if log else torch.softmax
        slot_preds = activation(self.word_outputs, dim=-1)
        if self.sentence_outputs is not None:
            intent_preds = activation(self.sentence_outputs, dim=-1)
        else:
            intent_preds = None
        return slot_preds, intent_preds


class MultiBERTForNLU(pl.LightningModule):
    """
    Multilingual BERT model for the NLU task.
    It has one head for intent detection and another one for slot filling.

    :param dataset: MultiATIS
        The MultiATIS dataset instance that
        will be used for training and validation
    :param config: Config
        The configuration parameters for loading
        the model and the tokenizer and for training
    """

    def __init__(self, dataset: MultiATIS, config: Config):
        super().__init__()
        self.cfg = config
        self.hparams = self.cfg.train.to_dict()

        # Load encoder
        self.bert = BertModel.from_pretrained(self.cfg.model.name_or_path)
        self.hidden_size = self.bert.config.hidden_size
        if self.cfg.train.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Initialize slot filling
        self.slot_filling_loss = SlotFillingLoss(
            dataset.num_slot_labels,
            coeff=self.cfg.train.slot_loss_coeff,
            ignore_index=self.cfg.dataset.ignore_index
        )
        self.dropout = nn.Dropout(self.cfg.train.dropout)
        self.token_clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.slot_filling_loss.num_labels),
        )
        self.val_slot_f1 = self._build_slot_metric(dataset, report=False)
        self.test_slot_f1 = self._build_slot_metric(dataset, report=True)

        # Initialize intent detection
        if self.cfg.train.do_intent_detection:
            self.intent_detection_loss = IntentDetectionLoss(dataset.num_intents)
            self.intent_clf = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.intent_detection_loss.num_labels),
            )
            self.val_intent_acc = Accuracy()
            self.test_intent_acc = Accuracy()

    def _build_slot_metric(self, dataset: MultiATIS, report: bool) -> SlotF1:
        return SlotF1(
            dataset.label_encoding,
            self.cfg.dataset.ignore_index,
            name_or_path=self.cfg.seqeval_path,
            compute_report=report
        )

    def _get_losses(
            self,
            model_output: ModelOutput,
            batch: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        slot_loss = self.slot_filling_loss(
            model_output.word_outputs,
            batch['slot_labels'],
            batch['attention_mask']
        )
        if self.cfg.train.do_intent_detection:
            intent_loss = self.intent_detection_loss(
                model_output.sentence_outputs,
                batch['intents']
            )
            total_loss = slot_loss + intent_loss
        else:
            intent_loss = None
            total_loss = slot_loss
        return total_loss, slot_loss, intent_loss

    def _log_loss(self, loss: torch.Tensor, stage: Text, name: Text, prog_bar: bool = True):
        self.log(
            f"{stage}_{name}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar
        )

    def _log_losses(
            self,
            loss: torch.Tensor,
            slot_loss: torch.Tensor,
            intent_loss: Optional[torch.Tensor],
            stage: Text
    ):
        self._log_loss(loss, stage, "loss")
        self._log_loss(slot_loss, stage, "slot_loss", prog_bar=False)
        if intent_loss is not None:
            self._log_loss(intent_loss, stage, "intent_loss", prog_bar=False)

    def forward(self, batch: Dict) -> ModelOutput:
        """
        Calculate a forward pass through the encoder and both classifiers

        :param batch: Batch
            Contains the tokenized input and both types of labels
        :return: ModelOutput
            Output embeddings and losses
        """
        output = self.bert(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask']
        )
        if self.cfg.train.do_intent_detection:
            sent_out = self.intent_clf(self.dropout(output.pooler_output))
        else:
            sent_out = None
        return ModelOutput(
            word_embeddings=output.last_hidden_state,
            sentence_embeddings=output.pooler_output,
            word_outputs=self.token_clf(self.dropout(output.last_hidden_state)),
            sentence_outputs=sent_out
        )

    def training_step(self, batch: Dict, batch_idx: int):
        loss, slot_loss, intent_loss = self._get_losses(self(batch), batch)
        # TODO this logging might not be safe in data parallel mode
        self._log_losses(loss, slot_loss, intent_loss, stage='train')
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        # Forward pass
        output: ModelOutput = self(batch)

        # Slot filling
        slot_preds, intent_preds = output.get_predictions()
        self.val_slot_f1(slot_preds, batch['slot_labels'])
        # TODO this logging might not be safe in data parallel mode
        self.log("dev_slot_f1", self.val_slot_f1, on_step=False, on_epoch=True)

        # Intent detection
        if self.cfg.train.do_intent_detection:
            self.val_intent_acc(intent_preds, batch['intents'])
            # TODO this logging might not be safe in data parallel mode
            self.log("dev_intent_acc", self.val_intent_acc, on_step=False, on_epoch=True)

        # Calculate loss
        loss, _, _ = self._get_losses(output, batch)
        self._log_loss(loss, "dev", "loss", prog_bar=False)

    def test_step(self, batch: Dict, batch_idx: int):
        # Forward pass
        output: ModelOutput = self(batch)

        # Accumulate performance
        slot_preds, intent_preds = output.get_predictions()
        self.test_slot_f1(slot_preds, batch['slot_labels'])
        if self.cfg.train.do_intent_detection:
            self.test_intent_acc(intent_preds, batch['intents'])

        # Calculate loss
        loss, _, _ = self._get_losses(output, batch)
        return loss

    def test_epoch_end(self, outputs: List[Any]) -> Dict[Text, Any]:
        slot_perf: Dict[Text, Any] = self.test_slot_f1.compute()
        perf = {
            "slot_filling_f1": slot_perf["f1"],  # torch.Tensor
            "slot_filling_report": slot_perf["report"]  # pandas.DataFrame
        }
        if self.cfg.train.do_intent_detection:
            perf["intent_accuracy"] = self.test_intent_acc.compute()  # torch.Tensor
        return perf

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
