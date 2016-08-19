import tensorflow as tf

def read_cifar10(filename_queue):

    #create class to store information for a single record
    class CIFAR10Record(object):
        pass
    result=CIFAR10Record()
    label_bytes=1
    result.height=32
    result.width=32
    result.depth=3
    image_bytes=result.height*result.width*result.depth
    total_bytes=image_bytes+label_bytes

    reader=tf.FixedLengthRecordReader(record_bytes=total_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes=tf.decode_raw(value,tf.uint8)
    result.label=tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.float32)

    record_img=tf.slice(record_bytes,[label_bytes],[image_bytes])
    record_img=tf.reshape(record_img,[result.depth,result.height,result.width])
    result.uint8image=tf.transpose(record_img,[1,2,0])

    return result

def inputs(eval_data,batch_size):
    if not eval_data:
        filenames=['./datasets/cifar-10-batches-bin/data_batch_%d.bin' %i for i in range(1,6)]
        num_per_epoch=50000
    else:
        filenames = ['./datasets/cifar-10-batches-bin/test_batch.bin']
        num_per_epoch=10000


    filename_queue=tf.train.string_input_producer(filenames)

    read_input=read_cifar10(filename_queue)
    image=tf.cast(read_input.uint8image,tf.float32)

    float_image=tf.image.per_image_whitening(image)

    min_frac_example_in_queue=0.4
    min_queue_example= int(num_per_epoch*min_frac_example_in_queue)

    return gen_batch(float_image,read_input.label,min_queue_example,batch_size)

def gen_batch(image,label,min_queue_examples,batch_size):
    num_preprocess_threads=2
    images,label_batch=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=num_preprocess_threads,capacity=min_queue_examples+3*batch_size,min_after_dequeue=min_queue_examples)
    tf.image_summary('images',images)
    return(images,tf.reshape(label_batch,[batch_size]))