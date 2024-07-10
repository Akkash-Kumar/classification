from keras.models import Sequential   #order layers sequentially

from keras.layers import Conv2D   #convolution layer,binary classification so 2D

from keras.layers import MaxPooling2D   #max pooling layer

from keras.layers import Flatten   #flatten process

from keras.layers import Dense   #fully connected layer

from tensorflow.keras.preprocessing.image import ImageDataGenerator    #preprocessing image to train

model = Sequential()    #initialise

model.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))  #add convolution layer with 32 neurons and with input layer

model.add(MaxPooling2D(pool_size = (2,2)))   #add pooling layer

model.add(Flatten())   #flatten process

model.add(Dense(units = 128,activation = 'relu'))  #add fully connected layer with 128 neurons

model.add(Dense(units = 1, activation = 'sigmoid'))  #add output layer with 1 neuron

model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])  #compile model


training_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)    #preprocess image for training

validation_datagen = ImageDataGenerator(rescale = 1./255)   #preprocess img for validation

training = training_datagen.flow_from_directory('Datasets/trainingData', target_size = (64,64) , batch_size = 5,class_mode = 'binary')  #get training data

validation = validation_datagen.flow_from_directory('Datasets/validationData', target_size = (64,64) , batch_size = 5,class_mode = 'binary')#get validation data

model.fit(training,steps_per_epoch = 10,epochs = 50,validation_data =  validation,validation_steps = 2)  #start training and validation



model_json = model.to_json()   

with open("model.json",'w') as json_file:   #save as model.json in folder, w- write

    json_file.write(model_json)    #save training in json file

model.save_weights("model.weights.h5")   #save weights as model.h5 in folder

print("Saved to folder")





 
