from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
import csv
import numpy as np
from scipy import ndimage
import cv2
from sklearn.utils import shuffle

# Load in the center camera data and steering angle
X_train = list()
Y_train = list()

with open('test.csv', 'rt') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		X_train.append(row[0])
		Y_train.append(row[3].strip())

print(Y_train)
print(X_train)

# resize images, currently 320x160
x_train = list()

for img in X_train:
	read_image = cv2.imread(img)
	resized_image = cv2.resize(read_image, (32, 16))
	x_train.append(np.array(resized_image.reshape( (1,) + resized_image.shape )))
#	x_train.append(resized_image)
y_train = list()

for y in Y_train:
	print(y)
	new_y = float(y)
	print(new_y)
	y_train.append(np.array(new_y))

print(x_train[0].shape)
print(x_train)
print(y_train)
# shuffle the data
x_train, y_train = shuffle(x_train, y_train)

# normalize the features ? do i need to do this?

# one hot pretty sure i don't need one hot because i only have one value

# Split into training and validation data sets, model.fit can do this for me

# Define the Model
model = Sequential()

# Model needs to output a single value, not a softmax either
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
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(43))

# Compile and train the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=10, validation_split=0.2)
history = model.fit(x_train, y_train, batch_size=128, nb_epoch=1, validation_split=0.15)

# Save model
with open("model.json", "w") as model_file:
	print(model.to_json(), file=model_file)

# Save weights
model.save_weights('model.h5')
