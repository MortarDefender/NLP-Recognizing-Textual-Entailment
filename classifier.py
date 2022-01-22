import transformers
import tensorflow as tf


class Classifier(tf.keras.Model):
    
    def __init__(self, model_name):
        
        super(Classifier, self).__init__()
        
        self.transformer = transformers.TFAutoModel.from_pretrained(model_name)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(rate=0.05)
        self.classifier = tf.keras.layers.Dense(3)

    def call(self, inputs, training=False):
        x = self.transformer(inputs, training=training)[0]        
        x = self.dropout(x, training=training)
        x = self.global_pool(x)
        
        return self.classifier(x)
