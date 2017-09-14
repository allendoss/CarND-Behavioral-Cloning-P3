import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

# Read the file path for each image
lines=[]
with open('D:/SDCND/Project_3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) #line here contains image file name for centre,left,right and, steering angle etc

#Extract the image name and steering angle for each image
images=[]
measurements=[]        
for line in lines:
    for i in range(3):
        source_path = line[i] #csv files store directory names with backslash instead of forward slash
        filename  = source_path.split("\\")[-1] #there are multiple directories each with a /, you only want the last part 'center_2017_08_08_16.jpg'
        current_path='D:/SDCND/Project_3/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    correction = 0.25
    measurement = float(line[3])
    measurements.append(measurement) # centre image steering angle
    #remember there is only one steering angle for all left,right,centre iamges
    #therefore manually modfiy the steering angle measurement
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)
    
    
# Keras only accepts numpy arrays
X_train = np.array(images) #list to numpy array
y_train = np.array(measurements)

print(X_train.shape)

# nVidia's End to End learning implemented using Keras
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#train data
model.fit(X_train,y_train, validation_split=0.2, shuffle=True, epochs=3) #reducing epoch to prevent overfitting
model.save('model.h5')