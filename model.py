from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Lambda, ELU
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
#	x_train.append(np.array(resized_image.reshape( (None,) + resized_image.shape )))
	new_image = resized_image[None, :, :, :]
	x_train.append(new_image)

y_train = list()

for y in Y_train:
	new_y = float(y)
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
ch, row, col = 3, 16, 32  # camera format

model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(row, col, ch),
		output_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

#model.compile(optimizer="adam", loss="mse")

# Compile and train the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=10, validation_split=0.2)
history = model.fit(x_train, y_train, batch_size=128, nb_epoch=1, validation_split=0.15)

# Save model
with open("model.json", "w") as model_file:
	print(model.to_json(), file=model_file)

# Save weights
model.save_weights('model.h5')
