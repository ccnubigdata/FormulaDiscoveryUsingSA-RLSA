import numpy as np
import matplotlib.pyplot as plt

y = np.load('loss.npy', allow_pickle=True)

ny = []
for i in range(len(y)-5):
    ny.append(sum(y[i:i+5])/5)

x = range(len(ny))

plt.plot(x, ny)
plt.show()


y = np.load('policy_loss.npy', allow_pickle=True)

ny = []
for i in range(len(y)-5):
    ny.append(sum(y[i:i+5])/5)

x = range(len(ny))

plt.plot(x, ny)
plt.show()


y = np.load('value_loss.npy', allow_pickle=True)

ny = []
for i in range(len(y)-5):
    ny.append(sum(y[i:i+5])/5)

x = range(len(ny))

plt.plot(x, ny)
plt.show()
