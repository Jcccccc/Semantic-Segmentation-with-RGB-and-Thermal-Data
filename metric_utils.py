import numpy as np
from sklearn.metrics import confusion_matrix

def iou(y_pre: np.ndarray, y_true: np.ndarray) -> 'dict':
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pre,
        labels=[0, 1, 2, 3, 4, 5])

    result_iou = [
        cm[i][i] / (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]) for i in range(len(cm))
    ]

    metric_dict = {}
    metric_dict['IOU_0'] = result_iou[0]
    metric_dict['IOU_1'] = result_iou[1]
    metric_dict['IOU_2']  = result_iou[2]
    metric_dict['IOU_3']  = result_iou[3]
    metric_dict['IOU_4'] = result_iou[4]
    metric_dict['IOU_5'] = result_iou[5]

    metric_dict['iou'] = np.mean(result_iou)
    metric_dict['accuracy'] = sum(np.diag(cm)) / sum(np.reshape(cm, -1))

    return metric_dict

