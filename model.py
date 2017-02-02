from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Lambda, ELU
import csv
import numpy as np
from scipy import ndimage
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# load in the center camera data and steering angle
X_train = list()
Y_train = list()

with open('driving_log.csv', 'rt') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		X_train.append(row[0])
		Y_train.append(row[3].strip())

# resize images, currently 320x160
x_train = list()

# I have a decent bit of ram on my computer so this is actually faster than doing incremental
# read in the data, crop the top off (region mask), and resize (for speed)
# then flip the image to double my dataset
for img in X_train:
	read_image = cv2.imread(img)
	crop_img = read_image[70:160, 0:320]
	new_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
	resized_image = cv2.resize(crop_img, (54, 15))
	flipped_image = resized_image.copy()
	flipped_image = cv2.flip(resized_image,1)
	x_train.append(resized_image)
	x_train.append(flipped_image)

y_train = list()

# convert to steering angle from string to float, include the flipped steering angle
for y in Y_train:
	new_y = float(y)
	flipped_y = -1. * new_y
	y_train.append(new_y)
	y_train.append(flipped_y)

# shuffle the data
x_train, y_train = shuffle(x_train, y_train)

# split the data
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.30, random_state = 0)
x_validate, x_test, y_validate, y_test = train_test_split(x_validate, y_validate, test_size = 0.50, random_state = 0)

# generator for model.fit_generator
def gen(images, labels, batch_size):
	start = 0
	end = start + batch_size
	n = len(images)
	while True:
		x_batch = np.array(images[start:end])
		y_batch = np.array(labels[start:end])
		start += batch_size
		end += batch_size
		if start >= n:
			start = 0
			end = batch_size
		yield (x_batch, y_batch)

# model (based off of comma.ai's, altered a bit to remove ELU and add relu's)
ch, row, col = 15, 54, 3  # camera format
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
		input_shape=(ch, row, col),
		output_shape=(ch, row, col)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation("relu"))
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation("relu"))
model.add(Dense(1))

#print out model
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

# train model
model.compile(optimizer="adam", loss="mse")
batch_size = 128
epochs = 5
model.fit_generator(
	gen(x_train, y_train, batch_size), 
	samples_per_epoch = len(x_train), 
	nb_epoch = epochs, 
	validation_data = gen(x_validate, y_validate, batch_size),
	nb_val_samples = len(x_validate)
)

# test model
test_score = model.evaluate(np.array(x_test), np.array(y_test))
print("\n")
print(test_score)
print(model.metrics_names)

# save model
with open("model.json", "w") as model_file:
	print(model.to_json(), file=model_file)

# save weights
model.save_weights('model.h5')
