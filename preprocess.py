import mnist
import scipy.misc as m
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
from trimap import trimap
import numpy as np

nodes_list = []

target_path = 'img/'
images = mnist.train_images()
img_re = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
labels = mnist.train_labels()



a=np.array(img_re[:200], dtype=np.float64)


triplets, weights = trimap(a, 2, 60, 10, 5, 1000)

for n in range(0,200):
    nodes_list.append({"id": n, "label": labels[n], "img": target_path + str(n) + '.png', "pixel": list(img_re[n])})
    img = m.toimage(m.imresize(images[n,:,:] * -1 + 256, 10.))
    m.imsave(target_path + str(n) + '.png', img)


triplet_list = []
weight_list = []

for n in range(triplets.shape[0]):
   triplet_list.append(list(triplets[n]))

for m in range(weights.shape[0]):
   weight_list.append(weights[m])






json_prep = {"nodes":nodes_list, "triplets":triplet_list, "weights":weight_list}

json_prep.keys()


json_dump = json.dumps(json_prep, indent=1, sort_keys=True)

filename_out = 'dists.json'
json_out = open(filename_out,'w')
json_out.write(json_dump)
json_out.close()
