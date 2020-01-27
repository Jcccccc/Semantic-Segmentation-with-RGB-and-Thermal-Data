import cv2
import numpy as np
import tensorflow as tf

def predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, prob: bool) -> np.ndarray:
    """
    Args:
        image: np.ndarray [size, size, 3]
        input_placeholder: tf.placeholder
        is_training_placeholder: tf.placeholder
        logits_prob_node: tf.Tensor [size, size, num_classes]
        sess: tf.Session
        prob: bool
    Returns:
        image_predict: np.ndarray [size, size, 5] if porb is True
                       np.ndarray [size, size] if prob is not True
    """
    #assert image.shape == (256, 256, 3), print(image.shape)
    # [256, 256, 3] -> [1, 256, 256, 3]
    #image = image.reshape((512, 512, 1))
    feed_dict = {input_placeholder: np.expand_dims(image, 0),
                 is_training_placeholder: False}
    image_predict_prob = sess.run(logits_prob_node, feed_dict=feed_dict)
    # [1, 256, 256, 5] -> [256, 256, 5]
    image_predict_prob = np.squeeze(image_predict_prob, 0)
    if prob:
        return image_predict_prob
    else:
        image_predict = np.argmax(image_predict_prob, -1)
        return image_predict

def rotate(x, angle, size):
    M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    x = cv2.warpAffine(x, M_rotate, (size, size))
    return x

def multi_scale_predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, multi: bool, size):
    """

    Args:
        image:
        input_placeholder:
        is_training_placeholder:
        logits_prob_node:
        sess:
        multi:

    Returns:
        np.ndarray [size, size]
    """

    kwargs = {
        'input_placeholder':input_placeholder,
        'is_training_placeholder':is_training_placeholder,
        'logits_prob_node':logits_prob_node,
        'sess':sess,
        'prob':True,
    }
    if multi:
        image_predict_prob_list = [
            predict(image=image, **kwargs)
        ]
        angle_list = [90, 180, 270]
        for angle in angle_list:
            image_rotate = rotate(image, angle, size)

            image_rotate_predict_prob = predict(image=image_rotate, **kwargs)
            image_predict_prob = rotate(image_rotate_predict_prob, -1 * angle, size)
            image_predict_prob_list.append(image_predict_prob)
        flip_list = [1, 0]
        for mode in flip_list:
            image_flip = cv2.flip(image, mode)

            image_flip_predict_prob = predict(image=image_flip, **kwargs)
            image_predict_prob = cv2.flip(image_flip_predict_prob, mode)
            image_predict_prob_list.append(image_predict_prob)
        final_predict_prob = sum(image_predict_prob_list) / len(image_predict_prob_list)
        return np.argmax(final_predict_prob, -1)
    else:
        kwargs['prob'] = False
        return predict(image, **kwargs)


def total_image_predict(ori_image_path: str,
                        input_placeholder: tf.placeholder,
                        is_training_placeholder: tf.placeholder,
                        logits_prob_node: tf.Tensor,
                        sess: tf.Session,
                        multi_scale = False
                        ) -> np.ndarray:

    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_COLOR)

    h_step = ori_image.shape[0] // 256
    w_step = ori_image.shape[1] // 256

    h_rest = -(ori_image.shape[0] - 256 * h_step)
    w_rest = -(ori_image.shape[1] - 256 * w_step)

    image_list = []
    predict_list = []
    for h in range(h_step):
        for w in range(w_step):
            image_sample = ori_image[(h * 256):(h * 256 + 256),
                           (w * 256):(w * 256 + 256), :]
            image_list.append(image_sample)
        image_list.append(ori_image[(h * 256):(h * 256 + 256), -256:, :])
    for w in range(w_step - 1):
        image_list.append(ori_image[-256:, (w * 256):(w * 256 + 256), :])
    image_list.append(ori_image[-256:, -256:, :])

    # predict
    for image in image_list:

        predict = multi_scale_predict(
            image=image,
            input_placeholder=input_placeholder,
            is_training_placeholder=is_training_placeholder,
            logits_prob_node=logits_prob_node,
            sess=sess,
            multi=multi_scale
        )
        predict_list.append(predict)

    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * 256:(h + 1) * 256,
            w * 256:(w + 1) * 256
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h * 256:(h + 1) * 256, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * 256):(w * 256 + 256)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[h_rest:, w_rest:] = predict_list[count_temp][h_rest:, w_rest:]
    return tmp





