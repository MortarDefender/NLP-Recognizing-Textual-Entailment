import nlp
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .dataSetUtils import getDevice, buildModel, tokenizeDataframe, loadDataSet, buildDataset


class Trainer:
    
    def __init__(self, trainDataSet, testDataSet, modelName = 'jplu/tf-xlm-roberta-large', epochs = 6, maxLength = 120, validation = "mnli+xnli"):

        self.epochs = epochs
        self.device = getDevice()
        self.maxLength = maxLength
        self.validation = validation
        self.auto = tf.data.experimental.AUTOTUNE  ##
        self.batchSize = 16 * self.device.num_replicas_in_sync
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        
        self.run(trainDataSet, testDataSet)

    def __loadXnli(self):
        
        result = []
        dataset = nlp.load_dataset(path='xnli')
        
        for key in dataset.keys():
            for record in dataset[key]:
        
                premise, hypothesis, label = record['hypothesis'], record['premise'], record['label']
                
                if premise and hypothesis and label in {0, 1, 2}:
                
                    for lang, translation in zip(hypothesis['language'], hypothesis['translation']):
                        premiseLang = premise.get(lang, None)
                    
                        if premiseLang is None:
                            continue
                        
                        result.append((premiseLang, translation, label, lang))

        return pd.DataFrame(result, columns = ['premise','hypothesis','label','lang_abv'])
    
    def __preprocess(self, data):
        
        return tokenizeDataframe(data, self.tokenizer, self.maxLength)
    
    def fit(self, trainDataset, validDataset, mnliFeatures, mnliLabels, xnliLabels, x_train):
        
        # fit parameters
        fit_params = dict(epochs = self.epochs, verbose = 2)
        nli_dataset = buildDataset(mnliFeatures, np.concatenate([mnliLabels, xnliLabels]), "train", self.auto, self.batchSize)
        
        # create TPU context
        with self.device.scope():
            model = buildModel(self.modelName, self.maxLength)
        
        if self.validation == "dataset":
            steps_per_epoch = len(x_train) // self.batchSize
            history = model.fit(
                trainDataset,
                steps_per_epoch = steps_per_epoch,
                validation_data = validDataset,
                **fit_params
            )
        elif self.validation == "mnli+xnli":
            steps_per_epoch = len(mnliFeatures) // self.batchSize
            history = model.fit(
                nli_dataset,
                steps_per_epoch = steps_per_epoch,
                validation_data = validDataset,
                **fit_params
            )
        return model, history
    
    def saveModel(self, model):
        # save weights
        model.save_weights("savedModel.h5")
        
    def plot(self, history, verbose = False):
        hist = history.history
        
        if verbose:
            print(max(hist['val_accuracy']))
        
        px.line(
            hist, x = range(1, len(hist['loss']) + 1), y = ['accuracy', 'val_accuracy'], 
            title = 'Model Accuracy', labels = {'x': 'Epoch', 'value': 'Accuracy'}
        )
    
    def run(self, trainDataSet, testDataSet):
        
        # load data
        train = pd.read_csv(trainDataSet) #TODO
        test = pd.read_csv(testDataSet) #TODO
        
        mnli = loadDataSet('glue', 'mnli')
        xnli = self.__loadXnli()  ##
        
        # tokenize  features, labels
        trainFeatures, trainLabels = self.__preprocess(train)
        testFeatures, testLabels = self.__preprocess(test)
        
        # project dataset validation 
        x_train, x_valid, y_train, y_valid = train_test_split(trainFeatures, trainLabels, test_size = 0.2, random_state = 2020)
        
        # nli datasets
        mnliFeatures, mnliLabels = self.__preprocess(mnli)
        xnliFeatures, xnliLabels = self.__preprocess(xnli)
        
        dataset = buildDataset(trainFeatures, trainLabels, "train", self.auto, self.batchSize)
        trainDataset = buildDataset(x_train, y_train, "train", self.auto, self.batchSize)
        validDataset = buildDataset(x_valid, y_valid, "valid", self.auto, self.batchSize)
        testDataset = buildDataset(testFeatures, None, "test", self.auto, self.batchSize)
        
        mnliFeatures += xnliFeatures
        
        model, history = self.fit(trainDataset, validDataset, mnliFeatures, mnliLabels, xnliLabels, x_train)
        self.saveModel(model)
        self.plot(history, True)
