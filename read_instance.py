import gzip
f = gzip.open('t10k-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 9999

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

import sys
img_index = int(sys.argv[1])

import matplotlib.pyplot as plt
image = np.asarray(data[img_index]).squeeze()
plt.imshow(image)
plt.show()
