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


def getUnbatchedDataset(trainDataSet, modelName, maxLength=64):
    
    if type(trainDataSet) == list:
        trainDataSet = {k: None for k in trainDataSet}
    
    trainDataSet = {k: v for k, v in trainDataSet.items() if k in raw_ds_mapping}    
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelName, use_fast=True)
    
    # This is a list of generators
    raw_datasets = [get_raw_dataset(x) for x in trainDataSet]
    
    nb_examples = 0

    labels = []    
    sentence_pairs = []
    
    for name in trainDataSet:
        raw_ds = get_raw_dataset(name)
        nb_examples_to_use = raw_ds_mapping[name][2]
        
        if trainDataSet[name]:
            nb_examples_to_use = min(trainDataSet[name], nb_examples_to_use)
        
        nb_examples += nb_examples_to_use
        
        n = 0
        
        for x in raw_ds:
            sentence_pairs.append((x['premise'], x['hypothesis']))
            labels.append(x['label'])
            n += 1
            if n >= nb_examples_to_use:
                break

    # `transformers.tokenization_utils_base.BatchEncoding` object -> `dict`
    r = dict(tokenizer.batch_encode_plus(batch_text_or_text_pairs = sentence_pairs, max_length = maxLength, padding = 'max_length', truncation = True))

    # This is very slow
    dataset = tf.data.Dataset.from_tensor_slices((r, labels))

    return dataset, nb_examples


def getBatchedTrainingDataset(dataset, nb_examples, batch_size = 16, shuffleBufferSize = 1, repeat = False):
    
    if repeat:
        dataset = dataset.repeat()
    
    if not shuffle_buffer_size:
        shuffle_buffer_size = nb_examples

    dataset = dataset.shuffle(shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset


def getPredictionDataset(dataset, batch_size = 16):
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
