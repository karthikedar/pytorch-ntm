# PyTorch Neural Turing Machine (NTM)

PyTorch implementation of [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (NTM).

An **NTM** is a memory augumented neural network (attached to external memory) where the interactions with the external memory (address, read, write) are done using differentiable transformations. Overall, the network is end-to-end differentiable and thus trainable by a gradient based optimizer.

The NTM is processing input in sequences, much like an LSTM, but with additional benfits: (1) The external memory allows the network to learn algorithmic tasks easier (2) Having larger capacity, without increasing the network's trainable parameters.

The external memory allows the NTM to learn algorithmic tasks, that are much harder for LSTM to learn, and to maintain an internal state much longer than traditional LSTMs.

## A PyTorch Implementation

This repository implements a vanilla NTM in a straight forward way. The following architecture is used:

![NTM Architecture](./images/ntm.png)

### Features
* Batch learning support
* Numerically stable
* Flexible head configuration - use X read heads and Y write heads and specify the order of operation
* **copy** and **repeat-copy** experiments agree with the paper

***

## Installation

The NTM can be used as a reusable module, currently not packaged though.

1. Clone repository
2. Install [PyTorch](http://pytorch.org/)
3. pip install -r requirements.txt

## Usage

Execute ./train.py

```
usage: train.py [-h] [--seed SEED] [--task {copy,repeat-copy}] [-p PARAM]
                [--checkpoint-interval CHECKPOINT_INTERVAL]
                [--checkpoint-path CHECKPOINT_PATH]
                [--report-interval REPORT_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed value for RNGs
  --task {copy,repeat-copy}
                        Choose the task to train (default: copy)
  -p PARAM, --param PARAM
                        Override model params. Example: "-pbatch_size=4
                        -pnum_heads=2"
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Checkpoint interval (default: 1000). Use 0 to disable
                        checkpointing
  --checkpoint-path CHECKPOINT_PATH
                        Path for saving checkpoint data (default: './')
  --report-interval REPORT_INTERVAL
                        Reporting interval
```
