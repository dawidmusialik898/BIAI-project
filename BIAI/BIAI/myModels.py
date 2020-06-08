import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from abc import ABC, abstractmethod

class ModelInterface(ABC):
	@abstractmethod
	def make1VGGModel():
		pass

	@abstractmethod
	def make2VGGModel():
		pass
	@abstractmethod
	def make3VGGModel():
		pass

#Basic models
class BaseModels(ModelInterface):

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
	def make1VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	def make2VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
	
	def make3VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay
class WeightDecay_Models(ModelInterface):
	def make1VGGModel():
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
	
	def make2VGGModel():
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
	
	def make3VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


#Models with weight decay and dropout
class WeighrDecay_Dropout_Models(ModelInterface):
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
	
	def make3VGGModel():
		model = keras.models.Sequential()
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
		model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.MaxPooling2D((2, 2)))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		return model
