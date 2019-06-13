import tensorflow as tf
import deepdish as dd
import numpy as np
# i = 0

# for example in tf.python_io.tf_record_iterator("../datasets/LLD-icon.tfrecord"):
#     result = tf.train.Example.FromString(example)
#     print(result)
#     if i >= 5:
#     	break
#     i += 1


mydata = dd.io.load("../datasets/LLD-icon.hdf5")
# print(mydata["labels"])
labels = mydata["labels"]["ae_grayscale"]

print(len(labels))
# data = mydata["data"]
# print(len(data))

# np.save("LLD-icon-labels.npy", labels)

lab = np.load("../datasets/LLD-icon-labels.npy")

print(len(lab))