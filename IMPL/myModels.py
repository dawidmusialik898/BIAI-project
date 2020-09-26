import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

from abc import ABC, abstractmethod

class ModelInterface(ABC):
	@staticmethod
	@abstractmethod
	def make1VGGModel():
		pass

	@staticmethod
	@abstractmethod
	def make2VGGModel():
		pass
	@staticmethod
	@abstractmethod
	def make3VGGModel():
		pass

#Basic models
class BaseModels(ModelInterface):

	@staticmethod
	def make1VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make2VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make3VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

#Models with dropout
class Dropout_Models(ModelInterface):
	@staticmethod
	def make1VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make2VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make3VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay
class WeightDecay_Models(ModelInterface):
	@staticmethod
	def make1VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make2VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make3VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=x))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay and dropout
class WeightDecay_Dropout_Models(ModelInterface):
	@staticmethod
	def make1VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make2VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	@staticmethod
	def make3VGGModel(x,y):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=x))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with various dropout
class VariousDropout_Models(ModelInterface):
	def make3VGGModel(x,y,z):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(z))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay and various dropout
class WeightDecay_VariousDropout_Models(ModelInterface):
	@staticmethod
	def make3VGGModel(x,y,z,r):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(z))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=r))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with batch normalization
class BatchNormalization_Models(ModelInterface):
	def make3VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with batch normalization and various dropout
class BatchNormalization_VariousDropout_Models(ModelInterface):
	def make3VGGModel(x,y,z):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(BatchNormalization())
		model.add(keras.layers.Dropout(z))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay and various dropout amd batch normalization
class WeightDecay_VariousDropout_BatchNormalization_Models(ModelInterface):
	@staticmethod
	def make3VGGModel(x,y,z,r):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(x))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(y))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=r))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(z))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=r))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

#Models with weight decay and batch normalization
class WeightDecay_BatchNormalization(ModelInterface):
	@staticmethod
	def make3VGGModel(x):
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x, input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=x))
		model.add(BatchNormalization())
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=x))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model