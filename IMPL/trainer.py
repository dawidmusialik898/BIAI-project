import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

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

def trainTest(trainer,datasets,date,dir,models):
	
		for datasetName, dataset in datasets.items():
			trainer.assignDataset(dataset)
			for modelName, model in models.items():
				print(f"training {modelName} model on {datasetName} dataset")
				print(f"./models/{date}_{modelName}.h5")
				history, acc = trainer.test(model)
				fileReport(history, acc, f"./"+dir+"/"+modelName)
				print(f"finished training {modelName} model on {datasetName} dataset")
				print("accuracy %.3f" % (acc * 100.0))
				model.save(f"./models/{date}_{modelName}_{datasetName}.h5")
				file = open(f"./"+dir+"/results.txt","a")
				file.write("Model: "+ modelName+ "     accuracy %.3f" % (acc * 100.0)+ "\n")
				file.close() 

#-----------------main-----------------#
def main():
	trainer = Trainer()

	cifar10 = Dataset()
	(cifar10.trainIn, cifar10.trainOut), (cifar10.testIn, cifar10.testOut) = keras.datasets.cifar10.load_data()
	cifar10 = cifar10.prepare()

	models = {}

	datasets = {}
	datasets["base"] = cifar10
	trainer.epochs = 200

	#dropouts
	models["dropout_3VGG_0.1"] = myModels.Dropout_Models.make3VGGModel(0.1)
	models["dropout_3VGG_0.2"] = myModels.Dropout_Models.make3VGGModel(0.2)
	models["dropout_3VGG_0.3"] = myModels.Dropout_Models.make3VGGModel(0.3)
	models["dropout_3VGG_0.4"] = myModels.Dropout_Models.make3VGGModel(0.4)
	models["dropout_3VGG_0.5"] = myModels.Dropout_Models.make3VGGModel(0.5)
	models["dropout_3VGG_0.6"] = myModels.Dropout_Models.make3VGGModel(0.6)

	
	#weight decay
	for x in [0.001,0.002, 0.003,0.004,0.005,0.006, 0.007, 0.008, 0.009, 0.01, 0.02]:
		models["weightDecay_l1_"+str(x)+"_3VGG"] = myModels.WeightDecay_Models.make3VGGModel(regularizers.l1(x))
		models["weightDecay_l2_"+str(x)+"_3VGG"] = myModels.WeightDecay_Models.make3VGGModel(regularizers.l2(x))
		models["weightDecay_l1l2_"+str(x)+"_3VGG"] = myModels.WeightDecay_Models.make3VGGModel(regularizers.l1_l2(x))
		
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"decay",models);
	models.clear();

	#weight decay + dropout
	for x in [0.001, 0.002, 0.003, 0.004, 0.005,0.006, 0.007]:
		for y in [0.1, 0.2, 0.3, 0.4]:
			#models["weightDecay_l1_"+str(x)+"_dropout_"+str(y)+"_3VGG_"] = myModels.WeightDecay_Dropout_Models.make3VGGModel(regularizers.l1(x),y)
			models["weightDecay_l2_"+str(x)+"_dropout_"+str(y)+"_3VGG_"] = myModels.WeightDecay_Dropout_Models.make3VGGModel(regularizers.l2(x),y)
			#models["weightDecay_l1l2_"+str(x)+"_dropout_"+str(y)+"_3VGG_"] = myModels.WeightDecay_Dropout_Models.make3VGGModel(regularizers.l1_l2(x),y)
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"decayDrop",models);
	models.clear();
	
	
	#varioous dropout
	for x in [0.1,0.2,0.3,0.4]:
			for y in [0.1,0.2,0.3,0.4]:
				for z in [0.1,0.2,0.3,0.4]:
					models["variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_3VGG"] = myModels.VariousDropout_Models.make3VGGModel(x,y,z)

	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"varDrop",models);
	models.clear();
	
	#various dropout + weight decay
	for x in [0.1,0.2,0.3,0.4]:
		for y in [0.1,0.2,0.3,0.4]:
			for z in [0.1,0.2,0.3,0.4]:
				for r in [0.001, 0.002, 0.003, 0.004, 0.005,0.006, 0.007]:
					models["variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay_l2_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_Models.make3VGGModel(x,y,z,regularizers.l2(r))
					#models["variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_Models.make3VGGModel(x,y,z,regularizers.l1_l2(r))
					#models["variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_Models.make3VGGModel(x,y,z,regularizers.l1(r))
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"decayVarDrop",models);
	models.clear();

	#batch normalization
	models["batchNormalization_3VGG"] = myModels.BatchNormalization_Models.make3VGGModel();
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNorm",models);
	models.clear();


	#various dropout + batch normalization
	for x in [0.1,0.2,0.3,0.4]:
			for y in [0.1,0.2,0.3,0.4]:
				for z in [0.1,0.2,0.3,0.4]:
					models["batchNormalization_variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_3VGG"] = myModels.BatchNormalization_VariousDropout_Models.make3VGGModel(x,y,z)	
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNormVarDrop",models);
	models.clear();

	#dropout + batch normalization
	for z in [0.1,0.2,0.3,0.4]:
		models["batchNormalization_dropout_"+str(x)+"_3VGG"] = myModels.BatchNormalization_VariousDropout_Models.make3VGGModel(x,x,x)
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNormDrop",models);
	models.clear();

	#batch normalization + weight decay
	for x in [0.001, 0.002, 0.003, 0.004, 0.005,0.006, 0.007]:
		#models["batchNorm_weightDecay_l1_"+str(x)+"_3VGG"] = myModels.WeightDecay_BatchNormalization.make3VGGModel(regularizers.l1(x))
		models["batchNorm_weightDecay_l2_"+str(x)+"_3VGG"] = myModels.WeightDecay_BatchNormalization.make3VGGModel(regularizers.l2(x))
		#models["batchNorm_weightDecay_l1l2_"+str(x)+"_3VGG"] = myModels.WeightDecay_BatchNormalization.make3VGGModel(regularizers.l1_l2(x))

	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNormDecay",models);
	models.clear();

	#dropout + batch normalization + weight decay
	models["weightDecay_batchNorma_variousDropout_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel();
	for x in [0.1,0.2,0.3,0.4]:
		for r in [0.001, 0.002, 0.003, 0.004, 0.005,0.006, 0.007]:
			#models["batchNorm_Dropout_"+str(x)+"_weightDecay_l1_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,x,x,regularizers.l1(r))
			models["batchNorm_Dropout_"+str(x)+"_weightDecay_l2_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,x,x,regularizers.l2(r))
			#models["batchNorm_sDropout_"+str(x)+"_weightDecay_l1l2_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,x,x,regularizers.l1_l2(r))
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNormDecayDrop",models);
	models.clear();

	#various dropout + batch normalization + weight decay
	for x in [0.1,0.2,0.3,0.4]:
		for y in [0.1,0.2,0.3,0.4]:
			for z in [0.1,0.2,0.3,0.4]:
				for r in [0.001, 0.002, 0.003, 0.004, 0.005,0.006, 0.007]:
					#models["batchNorm_variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay_l1_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,y,z,regularizers.l1(r))
					models["batchNorm_variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay_l2_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,y,z,regularizers.l2(r))
					#models["batchNorm_variousDropout_"+str(x)+"_"+str(y)+"_"+str(z)+"_weightDecay__l1l2_"+str(r)+"_3VGG"] = myModels.WeightDecay_VariousDropout_BatchNormalization_Models.make3VGGModel(x,y,z,regularizers.l1_l2(r))
	
	date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	trainTest(trainer,datasets,date,"batchNormDecayVarDrop",models);
	models.clear();
	
	
	#datasets = {}
	#datasets["base"] = cifar10
	#trainer.epochs = 100
	#date = datetime.now().strftime("%Y-%m-%d_%H-%M")
	#for datasetName, dataset in datasets.items():
	#	trainer.assignDataset(dataset)
	#	for modelName, model in models.items():
	#		print(f"training {modelName} model on {datasetName} dataset")
	#		print(f"./models/{date}_{modelName}.h5")
	#		history, acc = trainer.test(model)
	#		fileReport(history, acc, f"{datasetName}_{modelName}")
	#		print(f"finished training {modelName} model on {datasetName} dataset")
	#		print("accuracy %.3f" % (acc * 100.0))
	#		model.save(f"./models/{date}_{modelName}_{datasetName}.h5")

if __name__=="__main__":
	main()