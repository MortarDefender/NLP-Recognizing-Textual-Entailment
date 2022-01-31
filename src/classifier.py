import os
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score

from .train import Trainer
from .dataSetUtils import getDevice, buildModel, tokenizeDataframe, loadDataSet, buildDataset


class Classifier():
    
    def __init__(self, trainDataSet, testDataSet, outputFile, modelName = 'roberta-large-mnli', maxLength = 120):
        self.maxLength = maxLength
        self.modelName = modelName
        self.device = getDevice()
        self.auto = tf.data.experimental.AUTOTUNE
        self.batchSize = 16 * self.device.num_replicas_in_sync
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        
        self.run(trainDataSet, testDataSet, outputFile)  ##
    
    def __loadModel(self, modelFileName = "savedModel.h5"):
        
         with self.device.scope():
            self.model = buildModel(self.modelName, self.maxLength)
            self.model.load_weights(modelFileName)
    
    def __preprocess(self, data):
        
        # return self.__tokenizeDataframe(data, self.tokenizer, self.maxLength)
        return tokenizeDataframe(data, self.tokenizer, self.maxLength)
    
    def __preprocessQuery(self, query):
        punctuation = '[' + ''.join([c for c in string.punctuation if c != "'"]) + ']'
        
        resultingQurey = re.sub(punctuation, ' ', query.lower())
        return re.sub('[ ]{2,}', ' ', resultingQurey)
    
    def __searchBase(self, query, knowledgeBase):
        query = self.__preprocessQuery(query)
        return int(query in knowledgeBase)
    
    def predict(self, data):
        return self.model.predict(data, verbose = 0)
    
    def accuracy(self, features, trueLabels,extended = True, verbose = False):

        #if not extended:                                #2 class
         #   for i,l in enumerate(trueLabels):
          #      if l == 1:
           #         trueLabels[i] = 2

        dataset = buildDataset(features, trueLabels, "valid", self.auto, self.batchSize)
        predictions = np.argmax(self.model.predict(dataset), axis = 1)

        #if not extended:  # 2 class
         #   for i, l in enumerate(predictions):
          #      if l == 1:
           #         predictions[i] = 2

        
        if verbose:
            print(f"accuracy {accuracy_score(trueLabels, predictions):.4f}")
        
        return accuracy_score(trueLabels, predictions)
    
    def savePredictions(self, testDataset, output, outputFileName, extended = True):
        
        testPrections = self.predict(testDataset)
        predictions = testPrections.argmax(axis = 1)

        #if not extended:                           #2 class
         #   for i,l in enumerate(predictions):
          #      if l == 1:
           #         predictions[i] = 2

        output['prediction'] = predictions
        output.to_csv(outputFileName, index = False)
    
    def run(self, trainDataSet, testDataSet, outputFile, extended = True, verbose = False):
        
        if not os.path.isfile("savedModel.h5"):
            if verbose:
                print(f"saved model is missing.\nstart training on {trainDataSet}")

            Trainer(trainDataSet, testDataSet)
        
        # load data
        train = pd.read_json(trainDataSet)
        test = pd.read_json(testDataSet)
        output = pd.read_json(outputFile)
        
        # preprocess
        trainFeatures, trainLabels = self.__preprocess(train)
        testFeatures, testLabels = self.__preprocess(test)
        testDataset = buildDataset(testFeatures, None, "test", self.auto, self.batchSize)
        
        mnli = loadDataSet('glue', 'mnli')
        
        premises = pd.concat([train[['premise', 'lang_abv']], test[['premise', 'lang_abv']]])
        knowledgeBase = set(mnli['premise'].apply(self.__preprocessQuery))
        premises['mnli'] = premises['premise'].apply(lambda q: self.__searchBase(q, knowledgeBase))
        
        if verbose:
            print(f"fraction of train set english premises occurence in MNLI = {premises.loc[premises.lang_abv=='en', 'mnli'].mean() * 100}%") 
        
        self.__loadModel()
        self.accuracy(trainFeatures, trainLabels, extended, True)  #### accurace with the train data ??
        self.savePredictions(testDataset, output, 'submission.csv', extended)
