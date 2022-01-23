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

from .classifier import Classifier


def get_unbatched_dataset():
    pass

def get_batched_training_dataset():
    pass

def get_prediction_dataset():
    pass


class Trainer:
    
    def __init__(self, trainDataSet, modelName, maxLength = 64, replicaBatchSize = 16, 
                 predictionBatchSize = 64, shuffleBufferSize = 1):
        self.strategy = self.getStartegy()

        print('Number of replicas:', self.strategy.num_replicas_in_sync)             

        self.batch_size = replicaBatchSize * self.strategy.num_replicas_in_sync
        predictionBatchSize = predictionBatchSize * self.strategy.num_replicas_in_sync

        train_ds, self.nb_examples = get_unbatched_dataset(trainDataSet = trainDataSet, modelName = modelName, maxLength = maxLength)
        self.trainDataSet = get_batched_training_dataset(train_ds, self.nb_examples, batch_size = self.batch_size, 
                                                         shuffleBufferSize = shuffleBufferSize, repeat = True)
        
        valid_ds, self.nb_valid_examples = get_unbatched_dataset(dataSetNames = ['original valid'], modelName = modelName, maxLength = maxLength)
        self.validDataSet = get_prediction_dataset(valid_ds, predictionBatchSize)
        
        original_valid = None
        self.validLabels = next(iter(self.valid_ds.map(lambda inputs, label: label).unbatch().batch(len(original_valid))))
        
        test_ds, self.nb_test_examples = get_unbatched_dataset(dataSetNames = ['original test'], modelName = modelName, maxLength = maxLength)
        self.testDataSet = get_prediction_dataset(test_ds, predictionBatchSize)
        
        self.stepsPerEpoch = self.nb_examples // self.batch_size
    
    def getStartegy(self):
        strategy = None
        
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except ValueError:
            strategy = tf.distribute.get_strategy() # for CPU and single GPU
        
        return strategy
    
    def get_model(self, model_name, lr, verbose=False):

        with self.strategy.scope():

            model = Classifier(model_name)

            # False = transfer learning, True = fine-tuning
            model.trainable = True 

            if verbose:
                model.summary()

            # Instiate an optimizer with a learning rate schedule
            optimizer = tf.keras.optimizers.Adam(lr=lr)

            # Only `NONE` and `SUM` are allowed, and it has to be explicitly specified.
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.SUM)
            
            # Instantiate metrics
            metrics = {
                'train loss': tf.keras.metrics.Sum(),
                'train acc': tf.keras.metrics.SparseCategoricalAccuracy()
            }

            return model, loss_fn, optimizer, metrics
        
    def get_routines(self, model, loss_fn, optimizer, metrics):

        def train_1_step(batch):
            
            inputs, labels = batch
    
            with tf.GradientTape() as tape:

                logits = model(inputs, training = True)
                # Remember that we use the `SUM` reduction when we define the loss object.
                loss = loss_fn(labels, logits) / self.batch_size

            grads = tape.gradient(loss, model.trainable_variables)

            # Update the model's parameters.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))            
            
            # update metrics
            metrics['train loss'].update_state(loss)
            metrics['train acc'].update_state(labels, logits)

        @tf.function
        def dist_train_1_epoch(data_iter):
            """ Iterating inside `tf.function` to optimized training time. """
            for _ in tf.range(self.steps_per_epoch):
                self.strategy.run(train_1_step, args = (next(data_iter),))        

        @tf.function                
        def predict_step(batch):
            inputs, _ = batch
            
            logits = model(inputs, training = False)
            return logits

        def predict_fn(dist_test_ds):

            all_logits = []
            for batch in dist_test_ds:

                # PerReplica object
                logits = self.strategy.run(predict_step, args = (batch,))

                # Tuple of tensors
                logits = self.strategy.experimental_local_results(logits)

                # tf.Tensor
                logits = tf.concat(logits, axis = 0)

                all_logits.append(logits)

            # tf.Tensor
            logits = tf.concat(all_logits, axis = 0)

            return logits         
                
        return dist_train_1_epoch, predict_fn
        
    def train(self, train_name, model_name, epochs, verbose = False):
        self.strategy = self.getStartegy()

        print('Number of replicas:', self.strategy.num_replicas_in_sync)        
        
        model, loss_fn, optimizer, metrics = self.get_model(model_name, 1e-5, verbose=verbose)
        dist_train_1_epoch, predict_fn = self.get_routines(model, loss_fn, optimizer, metrics)
        
        train_dist_ds = self.strategy.experimental_distribute_dataset(self.trainDataSet)
        train_dist_iter = iter(train_dist_ds)
        
        dist_valid_ds = self.strategy.experimental_distribute_dataset(self.validDataSet)
        dist_test_ds = self.strategy.experimental_distribute_dataset(self.testDataSet)

        history = {}
        best_acc = 0.5
        
        for epoch in range(epochs):
            
            s = datetime.datetime.now()

            dist_train_1_epoch(train_dist_iter)

            # get metrics
            train_loss = metrics['train loss'].result() / self.stepsPerEpoch
            train_acc = metrics['train acc'].result()

            # reset metrics
            metrics['train loss'].reset_states()
            metrics['train acc'].reset_states()
                   
            print('epoch: {}\n'.format(epoch + 1))
            print('train loss: {}'.format(train_loss))
            print('train acc: {}\n'.format(train_acc)) 
                
            e = datetime.datetime.now()
            elapsed = (e - s).total_seconds()            
            
            logits = predict_fn(dist_valid_ds)

            valid_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(self.validLabels, logits, from_logits = True, axis = -1))
            valid_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(self.validLabels, logits))
            
            if valid_acc>best_acc:
                best_acc=valid_acc
                # save the model
                model.save_weights('best.h5')
                
            print('valid loss: {}'.format(valid_loss))
            print('valid acc: {}\n'.format(valid_acc))
            
            print('train timing: {}\n'.format(elapsed))
            
            history[epoch] = {
                'train loss': float(train_loss),
                'train acc': float(train_acc),
                'valid loss': float(valid_loss),
                'valid acc': float(valid_acc),                
                'train timing': elapsed
            }

            print('-' * 40)
        
        print('best acc:{}'.format(best_acc))
        model.load_weights('best.h5')
        logits = predict_fn(dist_test_ds)
        preds = tf.math.argmax(logits, axis = -1)
        
        submission = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')
        submission['prediction'] = preds.numpy()
        submission.to_csv(f'submission-{train_name}.csv', index = False)
        
        return history, submission,logits
