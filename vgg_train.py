import tensorflow as tf
import numpy as np
from load_data import load_training_data_into_tensors
import vgg
import time
import os

def create_placeholders(batch,pixel_width,pixel_height,channels,num_classes):

    inputs_placeholder=tf.placeholder("float",shape=[batch,pixel_width,pixel_height,channels],name='inputs_pl')
    labels_placeholder=tf.placeholder("float",shape=[batch,num_classes],name="labels_pl")

    return inputs_placeholder,labels_placeholder

def create_batch(i,inputs,labels):
    batch=100
    batch_inputs = np.array(inputs[((i%500)*batch):((i%500)*batch+batch),:,:,:],dtype='float')
    batch_labels = np.array(labels[((i%500)*batch):((i%500)*batch+batch),:],dtype='float')
    return batch_inputs, batch_labels


num_epochs=5
batch_size=100
max_iters=1000
image_width=32
image_height=32
chans=3
num_classes=10

data,labels= load_training_data_into_tensors()

#num_iters=data.get_shape().dims[0]

num_iters=num_epochs*50000/batch_size

if (num_iters>max_iters):
    iters=num_iters
else:
    iters=max_iters


def train(num_iters):
    train_data, train_labels = load_training_data_into_tensors()

    with tf.Graph().as_default():

        sess=tf.Session()
        batch_inputs,batch_labels=create_placeholders(100,32,32,3,10)

        logits=vgg.inference_vgg(batch_inputs)

        loss=vgg.loss(logits,batch_labels)
        train_op=vgg.training(loss,0.0001)
        saver = tf.train.Saver(tf.all_variables())
        #summary_op=tf.merge_all_summaries()
        #summary_writer=tf.train.SummaryWriter('log',sess.graph)

        sess.run(tf.initialize_all_variables())


        for iter in range(num_iters):
            start_time=time.time()
            inps,labs=create_batch(iters,train_data,train_labels)


            feed_dict={
                batch_inputs: inps,
                batch_labels: labs
            }


            _, total_loss = sess.run([train_op,loss],feed_dict=feed_dict)
            end_time=time.time()
            total_time=(end_time-start_time)

            if (iter%50)==0:

                print('Iteration: %d, Loss: %.2f, Iteration time: %.1f' %(iter,total_loss,total_time))

            #if (iter%50==0):
            #    summary_str=sess.run(summary_op)
            #    summary_writer.add_summary(summary_str,iter)

            if (iter%1000==0):
                checkpoint_path = os.path.join('./checkpoints', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=iter)

train(num_iters)


