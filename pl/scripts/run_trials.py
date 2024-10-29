#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Submits jobs to run all model training and evaluation on DeepGreen.
"""

from pathlib import Path

from submitter import main as submitter


def main():
    """
    Goal is 30 trials of each run
    300 epochs ADFA, 30 Epochs PLAID
    """
    models = ["cnnrnn", "lstm", "wavenet"]
    for i in range(30):
        for model in models:
            for single_model in range(1, 4):
                py_call = "python ../src/train_ensemble.py "
                job = f"{py_call} --epochs 300 --single_flag {single_model} --model {model} --data_set adfa --trial {i}"
                submitter(job, hours=10, job_name=f"{model}_a_{single_model}_{i:02d}")
                job = f"{py_call} --epochs 30 --single_flag {single_model} --model {model} --data_set plaid --trial {i}"
                submitter(job, hours=10, job_name=f"{model}_p_{single_model}_{i:02d}")


def save_scores(data_set):
    epoch = 300 if data_set == "adfa" else 30
    models = ["cnnrnn", "lstm", "wavenet"]

    for model in models:
        paths = Path(f"../trials/{model}_{data_set}_{epoch}").glob("model_*.ckpt")
        for path in paths:
            tokens = path.stem.split("_")
            new_path = path.parent / f"eval_{tokens[1]}_{tokens[2]}"
            if not Path(str(new_path) + ".npz").exists():
                py_call = f"python ../src/save_scores.py --path {str(path)} --data_set {data_set}"
                submitter(py_call, hours=3, job_name="eval", mem=64)


if __name__ == "__main__":
    main()
    save_scores("adfa")
    save_scores("plaid")
