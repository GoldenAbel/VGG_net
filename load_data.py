import numpy as np
import tensorflow as tf
import time


def unpickle(file):
    #imports the pickle file and returns a dictionary from the loaded file
    import cPickle
    fo = open(file,'rb')
    if (fo):
        dict=cPickle.load(fo)
        fo.close()
    else:
        print('no such file')
    return dict

def load_data():
    #loads all of the data/labels into a numpy matrix
    #data from Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
    train_data=[]
    train_labels=[]
    for i in range(1,6):
        data_dict=unpickle("./datasets/cifar-10-batches-py/data_batch_%d" %i)
        train_data.append(data_dict['data'])
        train_labels.extend(data_dict['labels'])

    train_data=np.vstack(train_data)

    return train_data,train_labels

def convert_to_rgb(data):
    data=tf.reshape(data,[-1,3,32,32],name='All_input_data')
    data=tf.transpose(data,[0,2,3,1])
    return data

def shuffle_data(data,labels):
    assert(len(data)==len(labels))
    shuffled_data=np.zeros_like(data)
    shuffled_labels=np.zeros_like(labels)
    shuffle_inds = np.random.permutation(len(data))
    for old_ind,new_ind in enumerate(shuffle_inds):
        shuffled_data[new_ind]=data[old_ind]
        shuffled_labels[new_ind]=labels[old_ind]
    return shuffled_data, shuffled_labels

def load_data_into_tensors():
    train_data,train_labels = load_data()
    train_data,train_labels= shuffle_data(train_data,train_labels)
    train_data=convert_to_rgb(train_data)
    train_labels=tf.convert_to_tensor(train_labels)
    return train_data,train_labels


time1=time.time()
load_data_into_tensors()
time2=time.time()
time_load=time2-time1
print('Time to load data: %f seconds.' %time_load)