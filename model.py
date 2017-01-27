from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
import csv

# Load in the center camera data and steering angle
# TODO CHANGE THE INPUT PIXEL SIZING OR CHANGE THE MODEL INPUT SIZING
# images are currently 320x160
X_input = list()
y_input = list()

with open('test.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		X_input.append(row[0])
		y_input.append(row[3])

print(y_input)
print(X_input)

# shuffle the data

# normalize the features ? do i need to do this?

# one hot

# Split into training and validation data sets, model.fit can do this for me

# Define the Model
#model = Sequential()

# Model needs to output a single value, not a softmax either
#model.add(Convolution2D(320, 5, 5, border_mode='valid', input_shape=(320, 160, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
#model.add(Convolution2D(14, 3, 3, border_mode='valid', input_shape=(14, 14, 6)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
#model.add(Flatten(input_shape=(14, 14, 6)))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dense(43))

# Compile and train the model
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, batch_size=128, nb_epoch=10, validation_split=0.2)

# Save model
#with open("model.json", "w") as model_file:
#	print(model.to_json(), file=model_file)

# Save weights
#model.save_weights('model.h5')