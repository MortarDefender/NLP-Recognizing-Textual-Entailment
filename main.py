import os
import numpy as np
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import plotly.express as px
import transformers
import tokenizers
import nlp
import datetime
import json

# os.environ["WANDB_API_KEY"] = "0" ## to silence warning



def run():
    print("run")


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    run()

