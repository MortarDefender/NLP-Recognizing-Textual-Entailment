import os
import nlp
import json
import random
import datetime
import tokenizers
import numpy as np
import transformers
import pandas as pd
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# os.environ["WANDB_API_KEY"] = "0" ## to silence warning

import sys

sys.path.append('../')

from src.train import Trainer



def run(trainDataSet, testDataSet, outputFileName = None, verbose = False):
    if verbose:
        print(f"start training on {trainDataSet}")
    
    # classifier = Training(trainDataSet, testDataSet, verbose)
    classifier = Trainer(trainDataSet, verbose)
    
    if verbose:
        print("finished training")
    
    # classifier.predict()
    predictionResults = classifier.predict(testDataSet, outputFileName)
    
    if predictionResults is not None:
        print(predictionResults)
    
    print(classifier.accuracy(testDataSet))


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    run()

