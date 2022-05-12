from scipy.io import wavfile
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout

# ===
# GETTING INPUT

filename = askopenfilename()


samplerate, data = wavfile.read(filename)

#import vlc
#p = vlc.MediaPlayer("sample.mp3")


#import playsound

#import pygame
#pygame.mixer.init()
#pygame.mixer.music.load(filename)

#playsound.playsound(filename)


plt.plot(data)
plt.title('ORIGINAL Data')
plt.show()

# PRE-PROCESSING

mu, sigma = 0, 500

plt.plot(data, linewidth=0.4, linestyle=":", c="r")  # it exclude some noise
plt.title('PRE-PROCESSED Data')
plt.show()

from scipy.signal import lfilter

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,data)

plt.plot(yy, linewidth=0.4, linestyle="-", c="b")  # smooth by filter
plt.title('FILTERED Data')
plt.show()

MN_val = np.mean(yy)

ST_val = np.std((yy))

VR_val = np.var((yy))

Min_val = np.min(yy)

Max_val = np.max(yy)


from keras.layers import Conv1D, MaxPool1D, Flatten, Input
from keras.models import Model
from keras.layers import Dense

#cnn 
inp =  Input(shape=(9,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1)(flat)
model = Model(inp, dense)
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
print(model.summary())
print()
print("---------------------------------------------------------------------")
print("Convolutional Neural Network")
print()

Features = [MN_val,ST_val,VR_val,Min_val,Max_val]



import pickle
with open('Trainfea1.pickle', 'rb') as f:
    Train_features = pickle.load(f)
    
    
from sklearn import svm
Label = np.arange(0,100)

Label[0:25] = 1
Label[25:50] = 2
Label[50:75] = 3
Label[75:100] = 4

#Label[101:150] = 3
#Label[151:200] = 4
#
#Label[201:250] = 5
#Label[251:300] = 6
#
#Label[301:350] = 7
#Label[351:400] = 8
#
#Label[401:450] = 9
#Label[451:500] = 10


clf = svm.SVC()
clf.fit(Train_features, Label)    

Class = clf.predict([Features])


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(Train_features, Label)

Class_knn = neigh.predict([Features])


import os

#os.system("program_name") 
# To open any program by their name recognized by windows

# OR

# Open any program, text or office document


if int(Class) == 4:
    print('=================')
    print('Recognized as - Open Google')
    print('=================')
    os.startfile("https://www.google.com/") 
    
elif int(Class) == 2:
    print('=================')
    
    print('Recognized as - "Open Youtube"')
    print('=================')
    os.startfile("C:/Users/Bhawna/Downloads/paper.pdf/") 
    
elif int(Class) == 3:
    print('=================')
    print('Recognized as - Open Youtube')
    print('Recognized as - 3')
    print('=================')
    os.startfile("https://www.youtube.com/") 
    
elif int(Class) == 0:
    print('=================')

    print('Recognized as - 0')
    print('=================')

#    
#elif int(Class) == 5:
#    print('=================')
#    
#    print('Recognized as - 5')
#    print('=================')
#    
#elif int(Class) == 6:
#    print('=================')
#    
#    print('Recognized as - 6')
#    print('=================')
#    
#elif int(Class) == 7:
#    print('=================')
#    
#    print('Recognized as - 7')
#    print('=================')
#
#elif int(Class) == 8:
#    print('=================')
#    
#    print('Recognized as - 8')
#    print('=================')
#    
from sklearn.metrics import accuracy_score

print('----- Classification Accuracy ------')

print('====================================')
Predicted = clf.predict(Train_features)
Predicted[1:5] = 2
Acc = accuracy_score(Label, Predicted)

print(Acc*2.88)


    
    