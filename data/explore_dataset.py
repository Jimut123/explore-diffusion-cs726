import numpy as np

from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# load the train dataset here -- helix dataset
data_helix_3D_train = np.load('helix_3D_train.npy')
print(data_helix_3D_train)
print(data_helix_3D_train.shape)


fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot the surface.
ax.scatter3D(data_helix_3D_train[:,0],data_helix_3D_train[:,1],data_helix_3D_train[:,2])
plt.show()



"""


# load the train dataset here
data_3d_sin_5_5_train = np.load('3d_sin_5_5_train.npy')
# print(data_3d_sin_5_5_train)
print(data_3d_sin_5_5_train[:,0])


# load the test dataset here
data_3d_sin_5_5_test = np.load('3d_sin_5_5_test.npy')
# print(data_3d_sin_5_5_test)
print(data_3d_sin_5_5_test[:,0])

fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot the surface.
ax.scatter3D(data_3d_sin_5_5_train[:,0],data_3d_sin_5_5_train[:,1],data_3d_sin_5_5_train[:,2],cmap=cm.jet)
# ax.scatter3D(data_3d_sin_5_5_train[:100,0],data_3d_sin_5_5_train[:100,1],data_3d_sin_5_5_train[:100,2],cmap='Greens')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot the surface.
ax.scatter3D(data_3d_sin_5_5_test[:,0],data_3d_sin_5_5_test[:,1],data_3d_sin_5_5_test[:,2],cmap=cm.jet)
# ax.scatter3D(data_3d_sin_5_5_train[:100,0],data_3d_sin_5_5_train[:100,1],data_3d_sin_5_5_train[:100,2],cmap='Greens')
plt.show()


data_3d_sin_5_5_bounds = np.load('3d_sin_5_5_bounds.npy')
print(data_3d_sin_5_5_bounds)
"""