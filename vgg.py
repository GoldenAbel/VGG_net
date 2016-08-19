import tensorflow as tf
import numpy as np

def conv_relu(inputs,name,filter_height,filter_width,channels,num_filters,stride):
    with tf.variable_scope(name) as scope:
        filter=tf.get_variable(name='W_conv',shape=[filter_height,filter_width,channels,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        conv=tf.nn.conv2d(inputs,filter,[1,stride,stride,1],padding='SAME')
        bias=tf.get_variable(name='b_conv',shape=[num_filters],initializer=tf.constant_initializer(0.0))
        z=conv+bias
        out = tf.nn.relu(z,name='activation')
        return out

def max_pool(inputs,name,pool_size,stride):
    with tf.variable_scope(name) as scope:
        out=tf.nn.max_pool(inputs,name='max_pool',ksize=[1,pool_size,pool_size,1],strides=[1,stride,stride,1],padding='SAME')
    return out

def fully_connected_relu(inputs,name,out_size):
    shape=inputs.get_shape()
    D=shape[1].value
    with tf.variable_scope(name) as scope:
        weight=tf.get_variable(name='W_affine',shape=[D,out_size],initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.get_variable(name='b_affine',shape=[out_size],initializer=tf.constant_initializer(0.0))
        z=tf.matmul(inputs,weight)+bias
        out=tf.nn.relu(z,name='activation_affine')
    return out


def inference_vgg(inputs):
    #building type D from paper

    inputs=tf.image.resize_images(inputs,224,224)
    #conv section 1
    #inputs batchx224x224x3
    conv1=conv_relu(inputs=inputs,name='conv1',filter_height=3,filter_width=3,channels=3,num_filters=64,stride=1)
    conv2=conv_relu(inputs=conv1,name='conv2',filter_height=3,filter_width=3,channels=64,num_filters=64,stride=1)

    #pool 1
    pool1=max_pool(inputs=conv2,name='pool1',pool_size=2,stride=2)

    #conv section 2
    conv3=conv_relu(inputs=pool1,name='conv3',filter_height=3,filter_width=3,channels=64,num_filters=128,stride=1)
    conv4=conv_relu(inputs=conv3,name='conv4',filter_height=3,filter_width=3,channels=128,num_filters=128,stride=1)

    #pool 2
    pool2=max_pool(inputs=conv4,name='pool2',pool_size=2,stride=2)

    #conv section 3
    conv5=conv_relu(inputs=pool2,name='conv5',filter_height=3,filter_width=3,channels=128,num_filters=256,stride=1)
    conv6=conv_relu(inputs=conv5,name='conv6',filter_height=3,filter_width=3,channels=256,num_filters=256,stride=1)
    conv7=conv_relu(inputs=conv6,name='conv7',filter_height=3,filter_width=3,channels=256,num_filters=256,stride=1)

    #pool 3
    pool3=max_pool(inputs=conv7,name='pool3',pool_size=2,stride=2)

    #conv section 4
    conv8=conv_relu(inputs=pool3,name='conv8',filter_height=3,filter_width=3,channels=256,num_filters=512,stride=1)
    conv9=conv_relu(inputs=conv8,name='conv9',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)
    conv10=conv_relu(inputs=conv9,name='conv_10',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)

    #pool 4
    pool4=max_pool(inputs=conv10,name='pool4',pool_size=2,stride=2)

    #conv section 5
    conv11=conv_relu(inputs=pool4,name='conv11',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)
    conv12=conv_relu(inputs=conv11,name='conv12',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)
    conv13=conv_relu(inputs=conv12,name='conv13',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)

    #pool 5
    pool5=max_pool(inputs=conv13,name='pool5',pool_size=2,stride=2)
    shape=pool5.get_shape()
    flatten_shape=shape[1].value*shape[2].value*shape[3].value
    fc_input=tf.reshape(pool5,[-1,flatten_shape])

    #fc
    fc1=fully_connected_relu(inputs=fc_input,name='fc1',out_size=4096)
    fc2=fully_connected_relu(inputs=fc1,name='fc2',out_size=4096)
    logits=fully_connected_relu(inputs=fc2,name='fc3',out_size=10)

    return logits



def loss(logits,labels):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels,name='trainloss'))

    return loss

def loss_readdata(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_ent_mean=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels,name='trainloss'))
    tf.add_to_collection('losses',cross_ent_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def training(loss, learning_rate):
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  # Create a variable to track the global step.
  train_op = optimizer.minimize(loss)
  return train_op

def training_readfromfile(loss, learning_rate,global_step):
  loss_op=tf.scalar_summary(loss.op.name,loss)

  learning_rate=tf.train.exponential_decay(learning_rate, global_step, 100, 0.98, staircase=False, name=None)
  tf.scalar_summary('learning_rate', learning_rate)
  # Create the gradient descent optimizer with the given learning rate.
  with tf.control_dependencies([loss_op]):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
      grads=optimizer.compute_gradients(loss)
  apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)


  # Create a variable to track the global step.
  with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
  return train_op

