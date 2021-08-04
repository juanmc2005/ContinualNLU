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

from typing import Tuple, Union
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import MultiBERTForNLU
from dataset import MultiATIS, ATISLanguage
from config import Config
from .base import LanguageTrainer


class MonolingualTrainer(LanguageTrainer):
    def __init__(self, dataset: MultiATIS, config: Config, log_dir: Union[Path, str]):
        super().__init__(config)
        self.dataset = dataset
        self.log_dir = log_dir

    def load_model(self, checkpoint_path: str, old_model: MultiBERTForNLU) -> MultiBERTForNLU:
        return MultiBERTForNLU.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=old_model.device,
            dataset=self.dataset,
            config=old_model.cfg
        )

    def train_on_language(
            self,
            model: MultiBERTForNLU,
            language: ATISLanguage,
            lang_code: str,
            pos: int,
            **kwargs):
        # Obtain data loaders
        train_loader = self.get_loader(language, "train")
        val_loader = self.get_loader(language, "dev")

        # Only save the best model per language according to the validation metric
        checkpoint_callback = ModelCheckpoint(
            monitor=f"dev_{self.config.validation_metric}",
            mode="min" if self.config.validation_metric == "loss" else "max",
            save_top_k=1,
        )
        # Log to a specific directory for this language, inside log_dir
        logger = TensorBoardLogger(self.log_dir, name=None, version=f"{pos}_{lang_code.upper()}")
        trainer = Trainer(
            logger=logger,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            max_epochs=self.config.epochs_per_lang,
            num_sanity_val_steps=0,
            **kwargs
        )

        trainer.fit(model, train_loader, val_loader)

        # Load the best checkpoint for next language
        model = self.load_model(trainer.checkpoint_callback.best_model_path, model)
        return trainer, model

    def train(
            self,
            model: MultiBERTForNLU,
            lang_code: str,
            pos: int,
            **kwargs
    ) -> Tuple[Trainer, MultiBERTForNLU]:
        return self.train_on_language(model, self.dataset[lang_code], lang_code, pos, **kwargs)
