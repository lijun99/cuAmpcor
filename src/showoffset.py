import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#data = np.loadtxt("data.txt").reshape((2048,2048))
data = np.memmap("DenseOffset.off", dtype=np.float32, mode='r', shape=(100,100,2))
data1=data[:,:,0]
print(data1)
plt.imshow(data1, cmap=cm.coolwarm)  
plt.show()

#plt.ylim([-2.2,0.2])
#plt.plot(data1[0,0:500])
#plt.show()
#pdata= data[:,:,1]
#nx, ny = 300, 300
#x = range(nx)
#y = range(ny)


#hf = plt.figure()
#ha = hf.add_subplot(111, projection='3d')

#X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
#ha.plot_surface(X, Y, pdata)

#plt.show()

