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
from typing import Union, Text

import yaml

from box import Box
from utils import fix_seed


class Config:
    """
    Dot-based access to configuration parameters saved in a YAML file.
    """
    def __init__(self, file: Union[Path, Text]):
        """
        Load the parameters from the YAML file.
        If no path are given in the YAML file for bert_checkpoint and seqeval, the corresponding objects will be load
        if used (needs an internet connection).
        """
        # get a Box object from the YAML file
        with open(str(file), 'r') as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        # manually populate the current Config object with the Box object (since Box inheritance fails)
        for key in cfg.keys():
            setattr(self, key, getattr(cfg, key))

        # resolve seqeval config into a name or a path
        seqeval_path = getattr(self, "seqeval_path", None)
        self.seqeval_path = seqeval_path if seqeval_path is not None else 'seqeval'

        self.dataset.path = Path(self.dataset.path)

        # Don't lowercase if the corresponding attribute is not defined in config.yml
        self.dataset.do_lowercase = getattr(self.dataset, 'do_lowercase', False)

        # Correct types in train (ex. lr = 5e-5 is read as string)
        for float_var in ["dropout", "learning_rate", "slot_loss_coeff"]:
            val = getattr(self.train, float_var)
            if type(val) != float:
                setattr(self.train, float_var, float(val))

        assert self.train.validation_metric in ["intent_acc", "slot_f1", "loss"], "Unrecognized validation metric"

        # Some attributes could not be defined in config.yml, set them as None
        self.train.num_workers = getattr(self.train, "num_workers", None)
        self.train.seed = getattr(self.train, "seed", None)

        # Fix seed if specified
        if self.train.seed is not None:
            fix_seed(self.train.seed)