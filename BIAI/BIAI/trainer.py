import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import myModels

class Dataset:
	def __init__(self, trainIn=None, trainOut=None, testIn=None, testOut=None):
		self.trainIn = trainIn
		self.trainOut = trainOut
		self.testIn = testIn
		self.testOut = testOut
	
	def prepare(self):
		dataset = Dataset()
		dataset.trainIn = self.trainIn.astype("float32") / 255.0
		dataset.testIn = self.testIn.astype("float32") / 255.0
		dataset.trainOut = to_categorical(self.trainOut)
		dataset.testOut = to_categorical(self.testOut)
		return dataset

class Trainer:
	def __init__(self):
		self.dataset = None
		self.epochs = 100
		self.batchSize = 64
		self.datasetPrepared = False

	def assignDataset(self, dataset):
		self.dataset = dataset
	
	def test(self, model):
		callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
		history = model.fit(
			self.dataset.trainIn,
			self.dataset.trainOut,
			epochs=self.epochs,
			batch_size=self.batchSize,
			validation_data=(self.dataset.testIn, self.dataset.testOut),
			verbose=9,
			callbacks=[callback]
		)
		_, acc = model.evaluate(self.dataset.testIn, self.dataset.testOut, verbose=2)

		return history, acc		

def fileReport(history, accuracy, filename):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='green', label='train')
	pyplot.plot(history.history['val_loss'], color='red', label='test')

	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='green', label='train')
	pyplot.plot(history.history['val_accuracy'], color='red', label='test')
	
	pyplot.tight_layout()

	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


#-----------------main-----------------#

trainer = Trainer()

cifar10 = Dataset()
(cifar10.trainIn, cifar10.trainOut), (cifar10.testIn, cifar10.testOut) = keras.datasets.cifar10.load_data()
cifar10 = cifar10.prepare()

models = {}


#basic models
models["base_1VGG"] = myModels.BaseModels.make1VGGModel()
models["base_2VGG"] = myModels.BaseModels.make2VGGModel()
models["base_3VGG"] = myModels.BaseModels.make3VGGModel()

#basic models with dropouts
models["dropout_1VGG"] = myModels.Dropout_Models.make1VGGModel()
models["dropout_2VGG"] = myModels.Dropout_Models.make2VGGModel()
models["dropout_3VGG"] = myModels.Dropout_Models.make3VGGModel()

#basic models with weight decay
models["decay_1VGG"] = myModels.WeightDecay_Models.make1VGGModel()
models["decay_2VGG"] = myModels.WeightDecay_Models.make2VGGModel()
models["decay_3VGG"] = myModels.WeightDecay_Models.make3VGGModel()

#basic models with weight decay and dropout
models["dropout_decay_1VGG"] = myModels.WeighrDecay_Dropout_Models.make1VGGModel()
models["dropout_decay_2VGG"] = myModels.WeighrDecay_Dropout_Models.make2VGGModel()
models["dropout_decay_3VGG"] = myModels.WeighrDecay_Dropout_Models.make3VGGModel()

datasets = {}
datasets["base"] = cifar10

trainer.epochs = 100
for datasetName, dataset in datasets.items():
	trainer.assignDataset(dataset)
	for modelName, model in models.items():
		print(f"training {modelName} model on {datasetName} dataset")
		history, acc = trainer.test(model)
		fileReport(history, acc, f"{datasetName}_{modelName}")
		print(f"finished training {modelName} model on {datasetName} dataset")
		print("accuracy %.3f" % (acc * 100.0))