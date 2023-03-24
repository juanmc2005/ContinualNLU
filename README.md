# Analyzing BERT Cross-lingual Transfer Capabilities in Continual Sequence Labeling

Companion repository for the paper ["Analyzing BERT Cross-lingual Transfer Capabilities in Continual Sequence Labeling"](https://aclanthology.org/2022.mmmpie-1.3.pdf).

This project contains all the scripts and information needed to reproduce the experiments presented in the paper.

## Citation

```bibtex
@inproceedings{coria-etal-2022-analyzing,
    title = "Analyzing {BERT} Cross-lingual Transfer Capabilities in Continual Sequence Labeling",
    author = "Coria, Juan Manuel  and
      Veron, Mathilde  and
      Ghannay, Sahar  and
      Bernard, Guillaume  and
      Bredin, Herv{\'e}  and
      Galibert, Olivier  and
      Rosset, Sophie",
    booktitle = "Proceedings of the First Workshop on Performance and Interpretability Evaluations of Multimodal, Multipurpose, Massive-Scale Models",
    month = oct,
    year = "2022",
    address = "Virtual",
    publisher = "International Conference on Computational Linguistics",
    url = "https://aclanthology.org/2022.mmmpie-1.3",
    pages = "15--25",
    abstract = "Knowledge transfer between neural language models is a widely used technique that has proven to improve performance in a multitude of natural language tasks, in particular with the recent rise of large pre-trained language models like BERT. Similarly, high cross-lingual transfer has been shown to occur in multilingual language models. Hence, it is of great importance to better understand this phenomenon as well as its limits. While most studies about cross-lingual transfer focus on training on independent and identically distributed (i.e. i.i.d.) samples, in this paper we study cross-lingual transfer in a continual learning setting on two sequence labeling tasks: slot-filling and named entity recognition. We investigate this by training multilingual BERT on sequences of 9 languages, one language at a time, on the MultiATIS++ and MultiCoNER corpora. Our first findings are that forward transfer between languages is retained although forgetting is present. Additional experiments show that lost performance can be recovered with as little as a single training epoch even if forgetting was high, which can be explained by a progressive shift of model parameters towards a better multilingual initialization. We also find that commonly used metrics might be insufficient to assess continual learning performance.",
}
```

## Installation

1) Create `conda` environment:

```shell
conda create -n ctcnlu python=3.8
conda activate ctcnlu
```

2) Install PyTorch (>=1.7.1) following the instructions of the [docs](https://pytorch.org/get-started/locally/#start-locally)

3) Install dependencies:
```shell
pip install -r requirements.txt
```

## Requirements

### Configuration file

Each experiment needs a YAML configuration file including various paths and hyper-parameters.

`example_config.yml` is an example of a configuration file.

### Dataset

To run the experiment you need to download the MultiATIS++ dataset.
Information about the dataset and how to download it can be found on this [github page](https://github.com/amazon-research/multiatis).

After downloading the dataset, the path to the directory containing the train, dev, test splits of each language (e.g.
`train_EN.tsv`) has to be indicated in the configuration file:
```yaml
dataset:
  # Path to TSV files for train, dev and test
  path: '/path/to/tsv/files'
```

### Offline mode (no internet connection)

If you try to run an experiment in **offline mode**, it will fail when trying to download the 
configured BERT, the associated tokenizer and the seqeval script.
To avoid this, do the following:
1. Pre-download the chosen model ('bert-base-multilingual-cased' is the one used in the paper) and the associated tokenizer (needs an internet connexion):
```python
from pathlib import Path
from utils import download_model

# Download and save locally both the model and the tokenizer from huggingface.co
download_model(save_dir=Path('/path/to/model/directory'), name='bert-base-multilingual-cased')
```
2. Get [seqeval.py](https://github.com/huggingface/datasets/blob/master/metrics/seqeval/seqeval.py) and save it locally. 
3. Modify `config.yml` with the paths to the dowloaded BERT model and `seqeval.py`:
```yaml
model:
  name_or_path: '/path/to/model/directory'

seqeval_path: '/path/to/seqeval.py'
```


## Run the experiments

To run an experiment, first add your `config.yml` to a new directory.

All experiments were run using seeds `0`, `100`, `200`, `300` and `400`.
You can set the seed by modifying `train.seed` in the config file.

For i.i.d. experiments the evaluation results are saved as a report to 
`<path_to_config.yml>/logs/slot_filling_report_test_<LANG>.csv`.
For continual experiments, the evaluation results are saved as a LxL performance matrix `P` (L=sequence length) to
`<path_to_config.yml>/logs/Sequence/slot_performance_test.csv`.

### Joint Transfer

To train the model on one language only (monolingual), set the `train.languages` parameter in the config file as a list
with the corresponding language code.

E.g. to train the model on English:
```yaml
train:
    languages: ['EN']
```

To train the model on a selection of languages at once (multilingual), fill the list with the associated language codes.

Then for both experiments, run the following:
```shell script
python train_iid.py --dir path/to/config.yml/directory
```

### Continual experiments

The model is trained on the sequence of languages specified in the config file under the `train.languages` parameter 
(describing the order in which the languages are learnt).

The list of sequences used to report our results can be found in `sequences_backward.lst` and 
`sequences_forward.lst`.

Summary of each continual experiment:

| experiment's name       | sequences used                     | value(s) used from `P`  |
| ----------------------- | ---------------------------------- | ----------------------- |
| Forward Transfer (FT)   | `sequences_forward_transfer.lst`   | `P_{L,L}`               |
| Backward Transfer (BT)  | `sequences_backward_transfer.lst`  | `P_{1,L}`               |
| Language Position FT    | `sequences_forward_transfer.lst`   | `P_{i, i}`              |
| Language Position BT    | `sequences_backward_transfer.lst`  | `P_{1, i}`              |
| Zero-shot               | `sequences_backward_transfer.lst`  | `P_{i, 1}`              |

To run a continual experiment:
```shell script
python train_continual.py --dir path/to/config.yml/directory
```


### Fast Recovery

To get the associated results, we used the model checkpoints obtained from the continual experiments on 
`sequences_backward_transfer.lst` sequences.

To run a fast recovery experiment:
```shell script
python train_iid.py --dir path/to/config.yml/directory --from_ckpt --epochs <num_epochs>
```

The checkpoint saved during training in the directory containing the config file of the continual experiment will be 
automatically found if it exists.


## License

```
MIT License

Copyright (c) 2021 Université Paris-Saclay
Copyright (c) 2021 Laboratoire national de métrologie et d'essais (LNE)
Copyright (c) 2021 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
