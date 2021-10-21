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

from pathlib import Path
import random
from re import match
from typing import Union, Text

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel


def download_model(save_dir: Path, name: Text):
    """
    Load pretrained tokenizer and model from huggingface.co and save them locally
    :param save_dir: Path
        Directory where the files are saved.
    :param name: Text
        Name of the BERT checkpoint to download (e.g. 'bert-base-multilingual-cased')
        The list of possible names can be found at https://huggingface.co/transformers/pretrained_models.html
    """
    print(f'Creating directory {save_dir} ... ', end='', flush=True)
    save_dir.mkdir(parents=True, exist_ok=False)
    print('OK')

    print(f"Getting model {name} ... ", end='', flush=True)
    # Load pretrained tokenizer and model from huggingface.co
    tokenizer = BertTokenizerFast.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    print('OK')

    print(f"Saving model {name} to {save_dir} ... ", end='', flush=True)
    # save the tokenizer and the model locally
    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))
    print('OK')


def as_path(path: Union[Path, Text]) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path


def load_performance(file: Union[Path, Text]) -> pd.DataFrame:
    """
    Load a performance matrix saved as CSV with column and row headers
    """
    return pd.read_csv(file, sep=',', index_col=0)


def fix_seed(seed: int = 42):
    """
    Set a fixed seed for the experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_dirs_in(path: Path, pattern):
    for d in path.iterdir():
        if d.is_dir() and match(pattern, d.name):
            yield d


def get_checkpoint_path(root: Path) -> Path:
    ckpt_path = root / "logs"
    last_lang_dirs = list(valid_dirs_in(ckpt_path, pattern=r"8_[A-Z]{2}"))
    assert len(last_lang_dirs) == 1, f"Found {len(last_lang_dirs)} last language directories, but only 1 should exist"
    ckpt_path = ckpt_path / last_lang_dirs[0] / "checkpoints"
    ckpt_files = []
    for path in ckpt_path.iterdir():
        if path.is_file() and path.name.endswith(".ckpt"):
            ckpt_files.append(path)
    assert len(ckpt_files) == 1, f"Found {len(ckpt_files)} checkpoints but only 1 should exist"
    return ckpt_files[0]
