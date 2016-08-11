import tensorflow as tf

def add_placeholders():
    self.input_placeholder=tf.placeholder(tf.int32,name='inputs',shape=[self.config.batch_size,32,32,3])
    self.labels_placeholder=tf.placeholder(tf.int32,name='labels',shape=[batch_size])
    self.dropout_placeholder=tf.placeholder(tf.float32)

def conv_relu(inputs,name,filter_height,filter_width,channels,num_filters,stride):
    with tf.variable_scope(name) as scope:
        filter=tf.get_Variable(name=scope+'W_conv',shape=[filter_height,filter_width,channels,num_filters],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(inputs,filter,[1,stride,stride,1],padding='SAME')
        bias=tf.get_variable(name=scope+'b',tf.zeros([num_filters]))
        z=conv+bias
        out = tf.nn.relu(z,name=scope)
        return out

def max_pool(inputs,name,pool_size,stride):
    out=tf.nn.max_pool(inputs,ksize=[1,pool_size,pool_size,1],strides=[1,stride,stride,1],padding='SAME',name=name)
    return out

def fully_connected_relu(inputs,name,in_size=None,out_size):
    if (in_size=None):
        in_size=inputs.get_shape()[-1].value

    with tf.Variable_scope(name) as scope:
        out=tf.contrib.layers.fully_connected(inputs,out_size,scope=scope,biases_initializer=tf.init_ops.zero_initializer)

     return out
def inference_vgg(inputs, labels=None):
    #building type D from paper
    #conv section 1
    #inputs 32x32
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
    conv10=conv_relu(inputs=conv9,name='conv10',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)

    #pool 4
    pool4=max_pool(inputs=conv9,name='pool4',pool_size=2,stride=2)

    #conv section 5
    conv11=conv_relu(inputs=pool4,name='conv11',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)
    conv12=conv_relu(inputs=conv11,name='conv12',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)
    conv13=conv_relu(inputs=conv12,name='conv13',filter_height=3,filter_width=3,channels=512,num_filters=512,stride=1)

    #pool 5
    pool5=max_pool(inputs=conv13,name='pool5',pool_size=2,stride=2)

    #fc 1
    fc1=fully_connected_relu(inputs=pool5,name='fc1',out_size=4096)
    fc2=fully_connected_relu(inputs=fc1,name='fc2',out_size=4096)
    fc3=fully_connected_relu(inputs=fc2,name='fc3',out_size=10)

    if (labels is not None):
        loss=tf.nn.softmax_cross_entropy_with_logits(fc3,labels,name='trainloss')



