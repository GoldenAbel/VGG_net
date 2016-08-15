import numpy as np
import tensorflow as tf
import time
#calling load data loads all the training labels and data into separate tensors and shuffles them
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

def load_train_data():
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

def convert_to_rgb(data,num_examples):
    data=np.reshape(data,[num_examples,3,32,32])
    data=np.transpose(data,[0,2,3,1])
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

def load_training_data():
    train_data,train_labels = load_train_data()
    train_data,train_labels= shuffle_data(train_data,train_labels)
    train_data=convert_to_rgb(train_data,50000)
    train_labels_one_hot=np.zeros((50000,10))
    train_labels_one_hot[np.arange(50000),train_labels]=1
    return train_data,train_labels_one_hot


def load_test_data():
    data_dict=unpickle("./datasets/cifar-10-batches-py/test_batch")

    test_data = data_dict['data']
    test_data=np.array(test_data)
    test_data=convert_to_rgb(test_data,10000)


    test_labels=data_dict['labels']
    test_labels=np.array(test_labels)
    #test_labels_one_hot=np.zeros((len(test_labels),10))
    #test_labels_one_hot[np.arange(len(test_labels)),test_labels]=1
    return test_data, test_labels


time1=time.time()
load_training_data()
time2=time.time()
time_load=time2-time1
print('Time to load data: %f seconds.' %time_load)