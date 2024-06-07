import numpy as np 

feature = np.arange(6, 21)
label = (feature * 3) + 4

print(feature)
print(label)

noise = (np.random.random([15]) * 4) - 2
label = label + noise 

print(noise)
print(label)
