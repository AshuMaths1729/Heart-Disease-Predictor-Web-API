from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy
import pandas as pd

dataset = numpy.loadtxt("data_mod.csv", delimiter="\t")

X = dataset[:,0:13]
Y = dataset[:,13]
X.shape
Y = Y.reshape(300,1)
Y.shape
Y
model = Sequential()
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=100)

model.save('trained_model.h5')
with open('model_summary.md','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))