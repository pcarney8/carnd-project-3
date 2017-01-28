from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Lambda, ELU
import csv
import numpy as np
from scipy import ndimage
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Load in the center camera data and steering angle
X_train = list()
Y_train = list()

with open('driving_log.csv', 'rt') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		X_train.append(row[0])
		Y_train.append(row[3].strip())

# resize images, currently 320x160
x_train = list()

for img in X_train:
	read_image = cv2.imread(img)
	resized_image = cv2.resize(read_image, (32, 16))
#	x_train.append(np.array(resized_image.reshape( (None,) + resized_image.shape )))
	x_train.append(resized_image)

y_train = list()

for y in Y_train:
	new_y = float(y)
	y_train.append(new_y)

# shuffle the data
x_train, y_train = shuffle(x_train, y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.15, random_state = 0)

# normalize the features ? do i need to do this?

# one hot pretty sure i don't need one hot because i only have one value
lb = preprocessing.LabelBinarizer()
y_one_hot = lb.fit_transform(np.array(y_train))

# Split into training and validation data sets, model.fit can do this for me

# Define the Model
# my model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(16, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(14, 3, 3, border_mode='valid', input_shape=(14, 14, 6)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten(input_shape=(14, 14, 6)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Dense(1))

# TODO: Compile and train the model

# Model needs to output a single value, not a softmax either
# commonai model
# ch, row, col = 3, 16, 32  # camera format
# model = Sequential()
# model.add(Lambda(lambda x: x/127.5 - 1.,
		# input_shape=(ch, row, col),
		# output_shape=(ch, row, col)))
# model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(512))
# model.add(Dropout(.5))
# model.add(ELU())
# model.add(Dense(1))

#model.compile(optimizer="adam", loss="mse")

# Compile and train the model
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=10, validation_split=0.2)
history = model.fit(np.array(x_train), y_one_hot, batch_size=128, nb_epoch=10, validation_split=0.15)

test_score = model.evaluate(x_test, y_test)
print(test_score)
print(model.metrics_names)

# Save model
with open("model.json", "w") as model_file:
	print(model.to_json(), file=model_file)

# Save weights
model.save_weights('model.h5')
