"""A `surprise`-based recommender implemented in SageMaker SKLearn framework

`scikit-surprise` is not part of `scikit-learn`, but is a common library used for building custom recommendation
engines. This file packages the module's popular "SVD" collaborative filtering algorithm for use with SageMaker via
the SageMaker SKLearn Estimator (since this is simpler than creating a custom container from scratch).
"""

# Python Built-Ins:
import argparse
from collections import namedtuple
import csv
from io import BytesIO, StringIO
import json
import os
import subprocess
import sys
from typing import List

# External Dependencies:
import numpy as np
import pandas as pd
subprocess.call([sys.executable, "-m", "pip", "install", "surprise"])
import surprise

# For type annotation purposes (for your info) only:
from surprise.prediction_algorithms.algo_base import AlgoBase as SurpriseAlgoBase


# Not-Worth-Making-Configuration:
ALGO_FILE_NAME = "model.surprise"

# Data model for a prediction request:
InferenceRequest = namedtuple("InferenceRequest", "uid iid")


def train(args):
    """Training script taking parsed command line / SageMaker variable arguments
    """
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n" +
                "This usually indicates that the channel ({}) was incorrectly specified,\n" +
                "the data specification in S3 was incorrectly specified or the role specified\n" +
                "does not have permission to access the data."
            ).format(args.train, "train")
        )
    train_df = pd.concat([
        pd.read_csv(file, engine="python") for file in input_files
    ])
    train_data = surprise.Dataset.load_from_df(
        train_df,
        surprise.Reader(line_format=u"user item rating", rating_scale=(1, 5))
    )
    algo = surprise.SVD()
    
    # Note: Quality metrics like this can be exposed to SageMaker if wanted, see:
    # https://sagemaker.readthedocs.io/en/stable/overview.html#training-metrics
    results = surprise.model_selection.cross_validate(
        algo,
        train_data,
        measures=("RMSE", "MAE"),
        verbose=True,
        cv=args.cross_validation_folds
    )
    
    # The main mission of our script is to train a model and then save it to file:
    algo.fit(train_data.build_full_trainset())
    surprise.dump.dump(os.path.join(args.model_dir, ALGO_FILE_NAME), algo=algo)

def model_fn(model_dir: str) -> SurpriseAlgoBase:
    """Deserialize and load previously fitted model
    """
    _, algo = surprise.dump.load(os.path.join(model_dir, ALGO_FILE_NAME))
    return algo

def input_fn(raw, content_type: str) -> List[InferenceRequest]:
    """Deserialize inference request data at runtime
    
    This implementation supports three formats:
    - CSV (as used in our example batch transform job)
    - raw numpy (the default for the SageMaker SKLearn real-time predictor)
    - A custom JSON format (to demonstrate a more web API-like alternative).
    
    In all formats, users can pass either a single record or a set of records in a request.
    """
    if content_type == "text/csv":
        stream = StringIO(raw)
        reader = csv.reader(stream)
        return [InferenceRequest(*datum) for datum in reader]
    elif content_type == "application/json":
        data = json.loads(raw)
        if type(data) == dict:
            return [InferenceRequest(uid=data["uid"], iid=data["iid"])]
        elif type(data) == list:
            return [InferenceRequest(uid=datum["uid"], iid=datum["iid"]) for datum in data]
        else:
            raise ValueError(
                "Got unsupported JSON data type '{}': Expected dict (single) or list (multiple)".format(
                    type(data)
                )
            )
        print(type(data))
        return [InferenceRequest(uid=data["uid"], iid=data["iid"])]
    elif content_type == "application/x-npy":
        stream = BytesIO(raw)
        arr = np.load(stream)
        dims = len(arr.shape)
        if dims == 1:
            return [InferenceRequest(*arr)]
        elif dims == 2:
            return [InferenceRequest(*row) for row in arr]
        else:
            raise ValueError(
                "Got {} dimensional input array: Expected 1 (single) or 2 (multiple)".format(
                    dims
                )
            )
    else:
        raise ValueError("Unexpected content_type '{}'".format(content_type))

def predict_fn(data: List[InferenceRequest], model: SurpriseAlgoBase) -> List[surprise.Prediction]:
    """Predict function called for inference"""
    return [model.predict(datum.uid, datum.iid, verbose=True) for datum in data]


# This section, executed only when the file is run as a script and not when it's imported as a module, loads
# environment variables and command line arguments and kicks off the training process:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cross-validation-folds', type=int, default=5)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    train(args)
