from deeplab_v3 import Deeplab_v3
from data_utils import DataSet


import cv2
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from color_utils import color_predicts
from predicts_utils import multi_scale_predict
from metric_utils import iou


class args:
    batch_size = 8
    lr = 1e-5
    num_steps = 20000
    test_display = 1000
    weight_decay = 5e-4
    model_name = 'thermal-512-1108'
    batch_norm_decay = 0.95
    test_img_dir = 'dataset_thermal/test/images'
    test_label_dir = 'dataset_thermal/test/labels'
    multi_scale = True
    gpu_num = 0
    pretraining = False
    ckpt_step = 2000

for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu_num

data_path_df = pd.read_csv('dataset/path_list.csv')
data_path_df = data_path_df.sample(frac=1)

dataset = DataSet(image_path=data_path_df['image'].values, 
                  label_path=data_path_df['label'].values,
                  data_type='thermal')

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 512, 512, 1], name='input_x')
label = tf.placeholder(tf.int32, [None, 512, 512])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
predicts = tf.argmax(logits, axis=-1, name='predicts')

variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')

restorer = tf.train.Saver(variables_to_restore)
# cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))

# https://arxiv.org/pdf/1807.11205.pdf
# weight_for_weightdecay = []
# for var in tf.trainable_variables():
#     if var.name.__contains__('weight'):
#         weight_for_weightdecay.append(var)
#         print(var.op.name)
#     else:
#         continue
with tf.name_scope('weight_decay'):
    l2_loss = args.weight_decay * tf.add_n(
         [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])


optimizer = tf.train.AdamOptimizer(learning_rate=lr)
loss = cross_entropy + l2_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables())
# for grad, var in grads:
#     if grad is not None:
#         tf.summary.histogram(name='%s_gradients' % var.op.name, values=grad)
#         tf.summary.histogram(name='%s' % var.op.name, values=var)
# gradients, variables = zip(*grads)
# gradients, global_norm = tf.clip_by_global_norm(gradients, 5)

apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=tf.train.get_or_create_global_step())
batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
train_op = tf.group(apply_gradient_op, batch_norm_updates_op)

saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# summary_op = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if args.pretraining:
        restorer.restore(sess, 'ckpts/resnet_v2_50/resnet_v2_50.ckpt')

    log_path = 'logs/%s/' % args.model_name
    model_path = 'ckpts/%s/' % args.model_name
    
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists('./logs'): os.makedirs('./logs')
    if not os.path.exists(log_path): os.makedirs(log_path)
        

    summary_writer = tf.summary.FileWriter('%s/' % log_path, sess.graph)

    learning_rate = args.lr
    losses, loss_steps = list(), list()
    for step in range(1, args.num_steps+1):
        if step == 15000 or step == 18000:
            learning_rate = learning_rate / 10
        x_tr, y_tr = dataset.next_batch(args.batch_size)

        loss_tr, l2_loss_tr, predicts_tr, _ = sess.run(
            fetches=[cross_entropy, l2_loss, predicts, train_op],
            feed_dict={
                image: x_tr,
                label: y_tr,
                model._is_training: True,
                lr: learning_rate})
        if step % 10 == 0:
            print(step, loss_tr, l2_loss_tr)
            losses.append(loss_tr)
            loss_steps.append(step)

        if (step in [50, 100, 200]) or (step > 0 and step % args.test_display == 0):

            all_tests_result, avg_tests_result = dict(), dict()
            for test_img_file in os.listdir(args.test_img_dir):
                test_img_path = os.path.join(args.test_img_dir, test_img_file)
                if os.path.isfile(test_img_path) and test_img_file.endswith('.jpg'):
                    test_img = cv2.imread(test_img_path, 0).reshape((512, 512, 1))
                    test_predict = multi_scale_predict(
                        image=test_img,
                        input_placeholder=image,
                        is_training_placeholder=model._is_training,
                        logits_prob_node=logits_prob,
                        sess=sess,
                        multi=args.multi_scale
                    )
                    test_label_file = test_img_file.split('.')[0] + '.png'
                    test_label_path = os.path.join(args.test_label_dir, test_label_file)
                    test_label = np.array(Image.open(test_label_path))

                    cv2.imwrite(filename='%spredict_color_%d.png' % (log_path, step),
                        img=color_predicts(img=test_predict))

                    result = iou(y_pre=np.reshape(test_predict, -1),
                         y_true=np.reshape(test_label, -1))
                    for key in sorted(result.keys()):
                        if np.isnan(result[key]) or result[key] != result[key]:
                            continue
                        if not key in all_tests_result:
                            all_tests_result[key] = [result[key]]
                        else:
                            all_tests_result[key].append(result[key])


            print("======================%d======================" % step)
            for key in sorted(all_tests_result.keys()):
                avg_tests_result[key] = sum(all_tests_result[key])/len(all_tests_result[key])
                offset = 40 - key.__len__()
                print(key + ' ' * offset + '%.4f' % avg_tests_result[key])

            test_summary = tf.Summary(
                value=[tf.Summary.Value(tag=key, simple_value=avg_tests_result[key]) for key in avg_tests_result.keys()]
            )
            summary_writer.add_summary(test_summary, step)
            summary_writer.flush()

        if step == 100 or step > 0 and step % args.ckpt_step == 0:
            model_path_step = saver.save(sess, model_path+'/'+args.model_name+'-%05d.ckpt'%step)
            print("Model saved in %s" % model_path_step)
            plt.plot(loss_steps, losses) 
            plt.xlabel('Steps') 
            plt.ylabel('Loss') 
            plt.title('Losses')
            plt.savefig(log_path+'losses.jpg')

