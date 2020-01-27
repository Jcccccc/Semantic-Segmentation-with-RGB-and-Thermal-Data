import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from predicts_utils import multi_scale_predict
from deeplab_v3 import Deeplab_v3
from color_utils import color_predicts
from metric_utils import iou


ckpt_path = './ckpts/rgbt-72f-wloss-1130/rgbt-72f-wloss-1130-20000.ckpt'
model_name = 'rgbt-72f-wloss-1130-BGR'
dataset = 'rgbt_balance'
model_type = 'rgbt'
test_list_path = dataset + '/test_list.csv'
result_dir = './results/' + model_name
os.mkdir(result_dir)

def restore_model(saved_file):
    # Restore the model using parameters dict
    variables = tf.global_variables()
    param_dict = {}
    for var in variables:
        var_name = var.name[:-2]
        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
        param_dict[var_name] = var
    saver = tf.train.Saver()
    saver.restore(sess, saved_file)

# Reset TF Graph
tf.reset_default_graph()
sess = tf.Session()

# Load BaseModel
model = Deeplab_v3(input_type=model_type)
if model_type == 'rgbt':
    image = tf.placeholder(tf.float32, [None, 512, 512, 4], name='input_x')
else:
    image = tf.placeholder(tf.float32, [None, 512, 512, 3], name='input_x')
label = tf.placeholder(tf.int32, [None, 512, 512])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
predicts = tf.argmax(logits, axis=-1, name='predicts')
restore_model(ckpt_path)

all_tests_result = {}
key_map = {'IOU_0': 0, 'IOU_1': 1, 'IOU_2': 2, 'IOU_3': 3, 'IOU_4': 4, 'IOU_5': 5}
with open(test_list_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for i, line in enumerate(reader):
        if i == 0:
            continue
        if model_type == 'rgbt':
            img_rgb = cv2.imread(line[0], cv2.IMREAD_COLOR)
            img_rgb = cv2.resize(img_rgb, (512, 512))
            img_thermal = cv2.resize(cv2.imread(line[1], 0), (512, 512)).reshape((512, 512, 1))
            img = np.concatenate((img_rgb, img_thermal), axis=2)
        else:
            img_rgb = cv2.imread(line[0], cv2.IMREAD_COLOR)
            img_rgb = cv2.resize(img_rgb, (512, 512))
            img = img_rgb
        test_predict = multi_scale_predict(
        	image=img,
        	input_placeholder=image,
        	is_training_placeholder=model._is_training,
            logits_prob_node=logits_prob,
            sess=sess,
            multi=False,
            size=512)
        g_truth = cv2.resize(np.array(Image.open(line[2])), (512, 512))
        result = iou(y_pre=np.reshape(test_predict, -1),
                     y_true=np.reshape(g_truth, -1))
        label_valid = np.unique(g_truth)
        for key in sorted(result.keys()):
            if key in key_map and not key_map[key] in label_valid:
                continue
            if np.isnan(result[key]) or result[key] != result[key]:
                continue
            if not key in all_tests_result:
                all_tests_result[key] = [result[key]]
            else:
                all_tests_result[key].append(result[key])
        test_label = color_predicts(img=g_truth)
        test_predict = color_predicts(img=test_predict)
        full_image = np.concatenate((img_rgb, test_label, test_predict), axis=1)
        out_name = line[0].split('/')[-1].split('.')[0]+'_predict.jpg'
        out_path = result_dir + '/' + out_name
        cv2.imwrite(out_path, full_image)
        print('Saved '+out_path)
avg_tests_result = {}
for key in sorted(all_tests_result.keys()):
    avg_tests_result[key] = sum(all_tests_result[key])/len(all_tests_result[key])
    offset = 40 - key.__len__()
    print(key + ' ' * offset + '%.4f' % avg_tests_result[key])

