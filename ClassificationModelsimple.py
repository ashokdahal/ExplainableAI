from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import numpy as np
FTP=tf.keras.metrics.TruePositives()
FFP=tf.keras.metrics.FalsePositives()
FFN=tf.keras.metrics.FalseNegatives()
class LandslideModel():
    def __init__(self):
        self.depth=12
    #Keras
    def DiceLoss(self,y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator
        
        # #flatten label and prediction tensors
        # TP=FTP(y_true, y_pred)
        # FP=FTP(y_true, y_pred)
        # FN=FFN(y_true, y_pred)
        # dice=2*((TP+smooth)/(TP+FP+TP+FN))

        # return 1 - dice

    def getclassificationModel(self,in_num=17,out_num=1):

        features_only=Input((in_num))

            
        x=layers.Dense(units=64,name=f'Sus_0',kernel_initializer='random_normal',bias_initializer='random_normal')(features_only)
        for i in range(1,self.depth+1):
            x=layers.Dense(units=64,name=f'Sus_{str(i)}',kernel_initializer='random_normal',bias_initializer='random_normal')(x)
            x= layers.BatchNormalization()(x)
            x=layers.Activation('relu')(x)
            #x= layers.Dropout(.3)(x)
        
        out_areaDen=layers.Dense(units=out_num,activation='sigmoid',name='sus')(x)
        self.model = Model(inputs=features_only, outputs=out_areaDen)

    def getOptimizer(self,opt=tf.keras.optimizers.Adam,lr=1e-3,decay_steps=10000,decay_rate=0.9):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_steps=decay_steps,decay_rate=decay_rate)
        self.optimizer = opt(learning_rate=lr_schedule)
   
    def compileModel(self,weights=None):
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0,1], threshold=0.5),tf.keras.metrics.AUC(),tf.keras.metrics.BinaryAccuracy()])
    
        