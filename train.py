from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

# Took only 3 classes so as to reduce training time
classes = ['table_tennis', 'football', 'swimming']
data = []
labels = []
dataset_path = ''
print('Collecting images ...')
for c in classes:
	imagePaths = list(paths.list_images(dataset_path + c + '/'))
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		# Preprocessing each image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))
		# Storing each image and its label in a list
		data.append(image)
		labels.append(c)

data = np.array(data)
labels = np.array(labels)

# One hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Splitting the data into train and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# Initializing data generator
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# Fine tuning using base model as RESNET
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freezing parameters for base layers
for layer in baseModel.layers:
	layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / 25)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print('Fitting model ...')
H = model.fit(
	x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=25)

predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

model_output_path = ''
lb_output_path = ''
model.save(model_output_path, save_format="h5")
f = open(lb_output_path, "wb")
f.write(pickle.dumps(lb))
f.close()