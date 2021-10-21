# Cross-lingual Transfer in Natural Language Understanding
Companion repository for the paper "A Study of Cross-lingual Transfer in Continual Natural Language Understanding"

This project contains all the scripts and information needed to reproduce the experiments presented in the paper.

## Installation

1) Create `conda` environment:

```shell
conda create -n acnlu python==3.8
conda activate acnlu
```

2) Install PyTorch (>= 1.7.1) following the instructions of the [docs](https://pytorch.org/get-started/locally/#start-locally)

3) Install dependencies:
```shell
pip install -r requirements.txt
```

## Requirements

### Configuration file

Each experiment needs a configuration file to be run. This file is a yaml file and includes paths and experiment 
parameters including the model hyperparameters.

`example_config.yml` is an example of a configuration file.

### Dataset

To run the experiment you need to download the MultiATIS++ dataset.
Information about the dataset and how to download it can be found on this [github page](https://github.com/amazon-research/multiatis).

After downloading the dataset, the path to the directory containing the train, dev, test splits of each languages (e.g.
`train_EN.tsv`) has to be indicated in the configuration file, as presented bellow: (excerpt from the config file)
```yaml
dataset:
  # Path to TSV files for train, dev and test
  path: '/path/to/tsv/files'
```

### Offline mode

If you try to run the experiments in **offline mode**, it will fail when trying to load the 
configured BERT, the associated tokenizer and the seqeval script.
To avoid this, do the following:
1. Pre-download the chosen model and the associated tokenizer (needs an internet connexion):
```python
from pathlib import Path
from utils import download_model

# Download and save locally both the model and the tokenizer from huggingface.co
download_model(save_dir=Path('/path/to/model/directory'), name='bert-base-multilingual-cased')
```
2. Get and save locally seqeval.py from https://github.com/huggingface/datasets/blob/master/metrics/seqeval/seqeval.py 
3. Modify `config.yml` with the paths of the dowloaded BERT checkpoint and `seqeval.py`: (excerpt from the config file)
```yaml
model:
  name_or_path: '/path/to/model/directory'

seqeval_path: '/path/to/seqeval.py'
```


## Run the experiments

For each experiment, first copy in a new directory `example_config.yml` and rename it `config.yml`.

All experiment were run using 5 different random seeds, that are `0`, `100`, `200`, `300` and `400`.
The results were then averaged and the standard deviation was computed.
To set the seed to use in an experiment, set the `train.seed` parameter in the config file.

Evaluation results are printed on the configured output.
For monolingual and multilingual experiments the results include multiple evaluation metrics per slot-type and averages 
metrics like the micro averaged F1-score (reported in the paper). The report is also automatically saved in 
`<path_to_config.ymal>/logs/slot_filling_report_test_<LANG>.csv`.

For continual experiments, the results consist in a LxL performance matrix P (L=sequence length) reporting the F1-score
of the model on each language after training on a language at each step of the training sequence.
The performance matrix is also automatically saved in
`<path_to_config.ymal>/logs/Sequence/slot_performance_test.csv`.

### Joint Transfer

Joint Transfer results can be obtained after running both monolingual and multilingual experiments.

To train the model on one language only (monolingual), set the `train.languages` parameters in the config file as a list
with only one string denoting the language you chose.

E.g. to train the model on the English data: (excerpt from the config file)
```yaml
train:
    languages: ['EN']
```

The model can also be trained at once on multiple languages mixed up together (multilingual). To train the model on a 
selection of languages, fill the list with the associated language codes.

Then for both experiments, run the following:
```shell script
python train_idd.py --dir path/to/config.yml/directory
```

### Continual experiments

The model is trained on the sequence of languages specified in the config file (the order of the elements in the list 
describes the order in which the languages are learnt).

Continual experiments were run on 3 different sequences beginning with the same language and 3 others ending with the same
language for each language of the dataset.
The list of sequences used to report our results can be found in `sequences_backward.lst` and 
`sequences_forward.lst`.

To run a continual experiment run the following:
```shell script
python train_continual.py --dir path/to/config.yml/directory
```

#### Continual Transfer

To get the results on **forward transfer**, we compare for each language the performance at the end of the training when the 
language is the **last one** in the sequence P_{L,L}, with monolingual and multilingual performances.
For these experiments only the sequences listed in `sequences_forward.lst` were used.

To get the results on **backward transfer**, we compare for each language the performance at the end of the training when the 
language is the **first one** in the sequence P_{1,L}, with monolingual and multilingual performances.
For these experiments only the sequences listed in `sequences_backward.lst` were used.

#### Language position

To get the associated results, we collected the performance matrices from the continual experiments on both 
`sequences_forward.lst` and `sequences_backward.lst` sequences and collated the individual performance P_{i,i} per 
training step.

#### Zero-shot Transfer

To get the associated results, we collated per sequence beginning with the same language, the performances of the first 
column of the performance matrices P_{i,1} obtained from the continual experiments on `sequences_backward.lst` 
sequences, and averaged the performance per languages.

### Fast Recovery

To get the associated results, we used the model checkpoints obtained from the continual experiments on 
`sequences_backward.lst` sequences, fine-tuned the model on the training dataset corresponding the the first language
of the sequence and averaged the results per language.

To run a fast recovery experiment run the following:
```shell script
python train_idd.py --dir path/to/config.yml/directory --from_ckpt --epochs 
```

The checkpoint saved during training in the directory containing the config file of the continual experiment will be 
automatically found if it exists.