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

from model import MultiBERTForNLU, MultiBERTTokenizer
from dataset import MultiATIS
from training import MonolingualTrainer, ContinualTrainer
from training.plugins import CheckpointCleaner
from config import Config
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
parser.add_argument('--gpus', type=int, default=1, help="Number of gpus to train on. Defaults to 1")
args = parser.parse_args()
args.dir = Path(args.dir)

# Load MultiATIS++ dataset
print()
print("Loading dataset... ", end="", flush=True)
config = Config(args.dir / "config.yml")
dataset = MultiATIS(config, MultiBERTTokenizer)
print("OK")

# Load multilingual BERT with NLU layers on top
print("Loading model...   ", end="", flush=True)
model = MultiBERTForNLU(dataset, config)
print("OK")

print("Available languages:", dataset.available_langs)

# Add custom plugins
plugins = []
if not config.train.keep_checkpoints:
    plugins.append(CheckpointCleaner(keep_last=False))

# Build monolingual trainer
log_dir = args.dir / "logs"
trainer = MonolingualTrainer(dataset, config, log_dir)
# Build continual learning trainer
trainer = ContinualTrainer(trainer, dataset, config, plugins)

print('Specified training sequence:', config.train.languages)
print()

# Train the model
trainer.train(model, gpus=args.gpus)
print("Done")
