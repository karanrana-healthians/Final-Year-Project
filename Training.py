


from scipy.signal import lfilter
from scipy.io import wavfile
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt 
import numpy as np

Trainfea1 = []
for ijkl in range(0,100):
    file = 'E:/voice_project/Train/SIG ('
    ext = ').wav'
    print(ijkl)
    templl = ijkl+1
    filename = file+str(templl)+ext
    
    
    samplerate, data = wavfile.read(filename)



    mu, sigma = 0, 500



    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b,a,data)


    MN_val = np.mean(yy)
    
    ST_val = np.std((yy))

    VR_val = np.var((yy))

    Min_val = np.min(yy)

    Max_val = np.max(yy)

    Features = [MN_val,ST_val,VR_val,Min_val,Max_val]
    
    
    Trainfea1.append(Features)
    
import pickle
with open('Trainfea1.pickle', 'wb') as f:
    pickle.dump(Trainfea1, f)      