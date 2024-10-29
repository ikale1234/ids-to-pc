#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Tools to evaluate previously trained models and save results to disk.
    Used by scripts/run_trials for automating jobs on DeepGreen.
"""

import argparse
import time
from itertools import chain
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_processing import get_data, load_nested_test

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.get_logger().setLevel("ERROR")


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Evaluate trained models and save scores.",
    )
    parser.add_argument(
        "--data_set",
        default="adfa",
        choices=["adfa", "plaid"],
        help="Data set to evaluate on.",
    )
    parser.add_argument(
        "--path",
        help="Location of model checkpoint.",
    )

    return parser


def get_scores(model, x_data, nll=False):
    """Calculates the probability of each sequence occurring.

    Parameters
    ----------
    model : tf.keras model
        Model to get probabilities from.
    x_data : obj
        Data to score.

    Returns
    -------
    Array of scores.

    """

    def predict_gen(model, x_data):
        it = iter(x_data)
        while True:
            try:
                data = next(it)
                preds = model(data).numpy()
                new_preds = []
                for pred, elm in zip(preds, data):
                    cutoff = np.argmax(elm == 0)
                    if cutoff != 0:
                        new_preds.append(pred[:cutoff])
                    else:
                        new_preds.append(pred)
                yield new_preds
            except StopIteration:
                break

    preds = []
    for x in predict_gen(model, x_data):
        preds.extend(x)
    probs = np.array([pred.max(axis=-1).prod(axis=-1) for pred in preds])

    if nll:
        return np.clip(-np.log2(probs), a_min=0, a_max=1e100)
    else:
        return probs


def save_scores(path, data_set):
    """Save scores for a given model to disk

    Parameters
    ----------
    path : str
        Path to model checkpoint
    data_set : {"adfa", "plaid"}

    Returns
    -------

    """
    path = Path(path)
    val, attack = load_nested_test(data_set)
    train_gen = get_data(data_set)[0]

    attack = list(chain(*attack))
    val = list(chain(*val))
    test = (
        tf.data.Dataset.from_tensor_slices(tf.ragged.constant(val + attack))
        .map(lambda x: x)
        .padded_batch(32, padded_shapes=(None,))
    )

    tokens = path.stem.split("_")
    new_path = path.parent / f"eval_{tokens[1]}_{tokens[2]}"
    if not Path(str(new_path) + ".npz").exists():
        model = load_model(str(path))
        t0 = time.time()
        scores = get_scores(model, test, nll=True)
        t1 = time.time()
        s = get_scores(model, train_gen.map(lambda x, y: x), nll=True)
        baseline = np.median(s)

        # scores, baseline, time
        np.savez_compressed(new_path, [scores, baseline, t1 - t0])
        print(new_path, t1 - t0)


if __name__ == "__main__":
    parser = create_parser()
    save_scores(**vars(parser.parse_args()))
