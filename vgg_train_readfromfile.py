import tensorflow as tf
import vgg
import time
import os
import load_binary_data as lbd


num_epochs=10
batch_size=100
max_iters=6000
image_width=32
image_height=32
chans=3
num_classes=10

num_iters=num_epochs*50000/batch_size

if (num_iters>max_iters):
    iters=num_iters
else:
    iters=max_iters


def train(num_iters):

    with tf.Graph().as_default():
        global_step=tf.Variable(0,trainable=False)

        batch_inputs,batch_labels=lbd.inputs(eval_data=False,batch_size=100)


        logits=vgg.inference_vgg(batch_inputs)

        loss=vgg.loss_readdata(logits,batch_labels)
        train_op=vgg.training_readfromfile(loss,0.1,global_step=global_step)
        summary_op=tf.merge_all_summaries()


        init=tf.initialize_all_variables()
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state('./checkpoints')
            if ckpt:
                print('Restoring from checkpoint')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint found')
                sess.run(init)
            print('initializing variables')


            print('initialized variables')
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.train.SummaryWriter('./logs', sess.graph)
            print('Beginning training')
            for iter in range(num_iters):

                start_time=time.time()
                step=sess.run(global_step)
                print(step)
                _, total_loss = sess.run([train_op,loss])
                end_time=time.time()
                total_time=(end_time-start_time)

                if (step%10)==0:

                    print('Iteration: %d, Loss: %.6f, Iteration time: %.1f' %(step,total_loss,total_time))

                if (step%10==0):
                    summary_str=sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)

                if (step%10==1):
                    checkpoint_path = os.path.join('./checkpoints', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

train(num_iters)


