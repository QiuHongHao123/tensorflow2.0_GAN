import numpy as np
import matplotlib.pyplot as plt


image=np.fromfile('./1.2.168.2101.441.100000073.20200313130928.130928_3D.raw',dtype='int16')
imageNp=np.reshape(image,[400,400,400])
testImage=imageNp[200]
