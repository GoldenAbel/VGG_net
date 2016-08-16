from load_data import load_test_data
import vgg
import tensorflow as tf
import numpy as np


def create_batch(i,inputs,labels):
    batch=100
    batch_inputs = np.array(inputs[((i)*batch):((i)*batch+batch),:,:,:],dtype='float')
    batch_labels = np.array(labels[((i)*batch):((i)*batch+batch)],dtype='float')
    return batch_inputs, batch_labels

def eval():

    test_data,test_labels=load_test_data()


    with tf.Graph().as_default() as g:

        test_inputs_placeholder=tf.placeholder(tf.float32,shape=[100,32,32,3],name='test_inputs')
        test_labels_placeholder=tf.placeholder(tf.int32,shape=[100],name='test_labels')
        logits=vgg.inference_vgg(test_inputs_placeholder)
        test_correct_op = tf.nn.in_top_k(logits,test_labels_placeholder,1)
        saver=tf.train.Saver()

        with tf.Session() as sess:
            #load most recent checkpoint
            ckpt=tf.train.get_checkpoint_state('./checkpoints')
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                global_step=int(global_step)
            curr_acc=0.0
            mean_acc=0.0
            for step in range((len(test_labels)/100)):
                test_batch_data,test_batch_labels=create_batch(step,test_data,test_labels)
                feed_dict={
                    test_inputs_placeholder:test_batch_data,
                    test_labels_placeholder:test_batch_labels
                }
                curr_correct=sess.run([test_correct_op],feed_dict=feed_dict)

                curr_acc=np.sum(curr_correct)/100.0
                mean_acc=((step)*mean_acc+curr_acc)/(step+1)
                print(mean_acc)
            print('Total mean accuracy is %f' %mean_acc)
eval()