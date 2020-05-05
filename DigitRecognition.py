from keras.datasets import mnist
from matplotlib import pyplot as plt
(trainx, trainy), (testx, testy) = mnist.load_data()

#visualising some of the images in the dataset
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(trainx[i],cmap ='gray')
plt.show()

#converting 3 channel images to a single channel image
trainx = trainx.reshape(trainx.shape[0],28,28,1)
testx = testx.reshape(testx.shape[0],28,28,1)

#preprocessing
from keras.utils import np_utils
trainx = trainx / 255
testx = testx / 255
trainy = np_utils.to_categorical(trainy)
testy = np_utils.to_categorical(testy)
num_classes = testy.shape[1]

#model
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D,Dense
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape =(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(trainx,trainy ,validation_data = (testx,testy),epochs = 10, batch_size=200)


#saving the model for future use
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")