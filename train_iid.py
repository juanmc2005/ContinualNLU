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

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import MultiATIS
from model import MultiBERTForNLU, MultiBERTTokenizer
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
parser.add_argument('--gpus', type=int, required=False, default=1, help="Number of gpus to train on")
args = parser.parse_args()
args.dir = Path(args.dir)

config = Config(args.dir / "config.yml")

languages = config.train.sequence
monolingual = len(languages) == 1

print("Loading dataset... ", end="", flush=True)
dataset = MultiATIS(config, MultiBERTTokenizer)
print("OK")

print("Loading model...   ", end="", flush=True)
model = MultiBERTForNLU(dataset, config)
print("OK")

# TRAINING
data_training = dataset[languages[0] if monolingual else 'all']
train_loader = data_training.get_loader(
    split='train',
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.num_workers
)
val_loader = data_training.get_loader(
    split='dev',
    batch_size=config.train.batch_size,
    shuffle=False,
    num_workers=config.train.num_workers
)

# Only save the best model according to the validation metric
validation_metric = config.train.validation_metric
checkpoint_callback = ModelCheckpoint(
    monitor=f"dev_{validation_metric}",
    mode="min" if validation_metric == "loss" else "max",
    save_top_k=1
)

# The checkpoints and logging files are automatically saved in save_dir
logger = TensorBoardLogger(save_dir=args.dir, name=None, version='logs')
trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=config.train.epochs_per_lang,
    num_sanity_val_steps=0,
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, train_loader, val_loader)

# EVALUATION
eval_split = 'test'
print(f"Evaluate the trained model on {eval_split}")

# Load the best checkpoint
print("Loading best checkpoint...   ", end="", flush=True)
best_ckpt = trainer.checkpoint_callback.best_model_path
model = MultiBERTForNLU.load_from_checkpoint(
    checkpoint_path=best_ckpt,
    map_location=model.device,
    dataset=dataset,
    config=model.cfg
)
print("OK")

if not monolingual:
    languages.append('all')

for lang in languages:
    # Load the split to test the trained model
    data_eval = dataset[lang] if lang != 'all' else data_training
    eval_loader = data_eval.get_loader(
        split=eval_split,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers
    )

    # Get slot-filling scores report with scores per slot types
    results = trainer.test(model, eval_loader, verbose=False)[0]
    report = results["slot_filling_report"]
    report.to_csv(Path(logger.log_dir) / f"slot_filling_report_{eval_split}_{lang.upper()}.csv")

# Delete checkpoint
Path(best_ckpt).unlink()
