# Introduction

this is a project which demo the usage of torch geometric

# Usage

## Install prerequisite packages

```shell
python3 -m pip install requirements.txt
```

## Train model

```shell
torchrun --nproc_per_node <data parallelism number> --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8088 train.py --input_csv dataset.csv
```
