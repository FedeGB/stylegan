import deepdish as dd
import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm # just a progressbar

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()

mydata = dd.io.load("LLD-icon.hdf5")
# print(mydata.keys())

# print(mydata['data'])

n_observations = len(mydata["labels"]["resnet"]["rc_32"])  # how many items are in your dataset
# loop through hdf5 of examples, save to tfrecord
with tf.io.TFRecordWriter(str('LLD-icon.tfrecords')) as writer:
    # for each example
    for exi in tqdm(range(n_observations)):
        # create an item in the datset converted to the correct formats (float, int, byte)
        example = serialize_example(
            {
                "labels": {
                    "data": mydata["labels"]["resnet"]["rc_32"][exi],
                    "_type": _int64_feature,
                },
                "data": {
                    "data": mydata["data"][exi].flatten().tobytes(),
                    "_type": _bytes_feature,
                },
            }
        )
        # write the defined example into the dataset
        writer.write(example)