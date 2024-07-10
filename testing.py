from keras.models import model_from_json    #to load model.json file

import numpy    #change image dimension

from keras.preprocessing import image   #preprocess image


json_file = open('model.json','r')   #open model.json file in folder, r- read

loaded_model_json = json_file.read()

json_file.close()

model = model_from_json(loaded_model_json)   #load json file into model

model.load_weights("model.weights.h5")

print("Loaded from folder ")

def classify(img_file):   #function define

    img_name = img_file

    test_image = image.load_img(img_name,target_size = (64,64))    #load image

    test_image = image.img_to_array(test_image)   #convert to array

    test_image = numpy.expand_dims(test_image,axis = 0)  #dimension expand

    result = model.predict(test_image)   #predict image

    if result[0][0] == 1:

        answer = 'Orange'   #if op is 1 then it is orange

    else:

        answer = 'Watermelon'   #else watermelon

    print(answer , img_name)   #print answer and image name

import os   #to access file directories

path = "C:/Python AI projects/Deep learning projects/project - 3 - image classification/Datasets/testData"    #test data path

files = []     #initialise files

for r,d,f in os.walk(path):    #root,directory,files

    for file in f:

        if '.jpg' in file:   #check whether file is jpeg

            files.append(os.path.join(r,file))   #add file in files array


for f in files:   #looping files in array

    classify(f)   #call function classify

    print('\n')




    

    






