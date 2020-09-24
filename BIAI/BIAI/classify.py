import tensorflow as tf
from tensorflow import keras
import argparse
import numpy as np

def predictionToLabel(prediction):
    imageClass = np.argmax(prediction)
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    return labels[imageClass]

def classify(args):
    imagePath = args.image
    modelPath = args.model

    image = keras.preprocessing.image.load_img(imagePath)
    model = keras.models.load_model(modelPath)

    if image is None:
        print(f"could not load image {imagePath}")
        return
    if model is None:
        print(f"could not load model {modelPath}")
        return

    imageArray = keras.preprocessing.image.img_to_array(image) * (1.0 / 255.0)
    imageArray = imageArray.reshape((1, imageArray.shape[0], imageArray.shape[1], imageArray.shape[2]))
    print(imageArray.dtype)
    print(imageArray.shape)

    prediction = model.predict(imageArray)
    print(prediction)
    label = predictionToLabel(prediction)
    print(label)

def main():
    parser=argparse.ArgumentParser(description="Classify a CIFAR-10 image")
    parser.add_argument("-image", help="image to classify", dest="image", type=str, required=True)
    parser.add_argument("-model", help="model used for classification", dest="model", type=str, required=True)
    parser.set_defaults(func=classify)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()