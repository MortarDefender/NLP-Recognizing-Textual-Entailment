import nlp
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D


def getDevice(verbose = False):
        
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        device = tf.distribute.experimental.TPUStrategy(tpu)
        
        if verbose:
            print("Init TPU device")

    except ValueError:
        device = tf.distribute.get_strategy() # for CPU and single GPU
        
        if verbose:
            print("Init CPU/GPU device")
        
    return device


def buildModel(modelName, maxLength, mode = "avg_pooling"):
    
    inputs = Input(shape = (maxLength,), dtype = tf.int32, name = "input_ids")
    encoder = TFAutoModel.from_pretrained(modelName)
    encoderOutput = encoder(inputs)[0]
    
    # convert transformer encodings to 1d-vector
    if mode == "cls":
        features = encoderOutput[:, 0, :] # using first token as encoder feature map
    elif mode == "avg_pooling":
        features = GlobalAveragePooling1D()(encoderOutput)
    elif mode == "max_pooling":
        features = GlobalMaxPooling1D()(encoderOutput)
    else:
        raise NotImplementedError
    
    # 3-class softmax
    out = Dense(3, activation = 'softmax')(features)
    
    # define model
    model = Model(inputs = inputs, outputs = out)
    model.compile(
        Adam(lr = 1e-5), 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy']
    )

    return model


def tokenizeDataframe(data, tokenizer, maxLength):
    
    text = data[['premise', 'hypothesis']].values.tolist()  #TODO
    encoded = tokenizer.batch_encode_plus(text, padding = True, max_length = maxLength, truncation = True)
    
    labels = data.label.values if 'label' in data.columns else None
    features = encoded['input_ids']
    
    return features, labels


def loadDataSet(dataSetPath, dataSetName = None, useValidation = True):  # mnli, snli
    
    result = []
    keys = ['train', 'validation'] if useValidation else ['train']
    
    #if dataSetName is None:
     #   dataset = nlp.load_dataset(path = dataSetPath)
    #else:
     #   dataset = nlp.load_dataset(path = dataSetPath, name = dataSetName)
    dataset = pd.read_json(trainDataSet)

    for key in keys:
        for record in dataset[key]:
            
            premise, hypothesis, label = record['premise'], record['hypothesis'], record['label']
            
            if premise and hypothesis and label in {0, 1, 2}:
                result.append((premise, hypothesis, label, 'en'))
    
    return pd.DataFrame(result, columns = ['premise', 'hypothesis', 'label', 'lang_abv'])


def buildDataset(features, labels, mode, auto, batchSize):
    
    if mode == "train":
        dataset = (
            tf.data.Dataset.from_tensor_slices((features, labels)).repeat().shuffle(2048)
            .batch(batchSize).prefetch(auto)
        )
        
    elif mode == "valid":
        dataset = (
            tf.data.Dataset.from_tensor_slices((features, labels))
            .batch(batchSize).cache().prefetch(auto)
        )
        
    elif mode == "test":
        dataset = (
            tf.data.Dataset.from_tensor_slices(features).batch(batchSize)
        )
        
    else:
        raise NotImplementedError
    
    return dataset
