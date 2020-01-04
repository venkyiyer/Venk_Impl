import argparse
import math
import cv2

import tensorflow as tf
import numpy as np
import os
from collections import namedtuple
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))


Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize','seg_gt', 'seg_gt_to_compare'])
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])


def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

def str2bool(v):
    """
    Convert a string to a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def abs2prop(xmin, xmax, ymin, ymax, imgsize):
    """
    Convert the absolute min-max box bound to proportional center-width bounds
    """
    width   = float(xmax-xmin)
    height  = float(ymax-ymin)
    cx      = float(xmin)+width/2
    cy      = float(ymin)+height/2
    width  /= imgsize.w
    height /= imgsize.h
    cx     /= imgsize.w
    cy     /= imgsize.h
    return Point(cx, cy), Size(width, height)


def prop2abs(center, size, imgsize):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)

def box_is_valid(box):
    for x in [box.center.x, box.center.y, box.size.w, box.size.h]:
        if math.isnan(x) or math.isinf(x):
            return False
    return True

def normalize_box(box):
    if not box_is_valid(box):
        return box

    img_size = Size(1000, 1000)
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    xmin = max(xmin, 0)
    xmax = min(xmax, img_size.w-1)
    ymin = max(ymin, 0)
    ymax = min(ymax, img_size.h-1)

    # this happens early in the training when box min and max are outside
    # of the image
    xmin = min(xmin, xmax)
    ymin = min(ymin, ymax)

    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
    return Box(box.label, box.labelid, center, size)

def draw_box(img, box, color):
    img_size = Size(img.shape[1], img.shape[0])
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    img_box = np.copy(img)
    cv2.rectangle(img_box, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img_box, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_box, box.label, (xmin+5, ymin-5), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(img_box, alpha, img, 1.-alpha, 0, img)
#------------------------------------------------------------------------------

def one_hot_encode(image):
    color = [[0, 0, 0],  # black-background,
             [0, 0, 128],  # maroon-aeroplane,
             [0, 128, 0],  # darkgreen-bicycle
             [0, 128, 128],  # gold-bird,
             [128, 0, 0],  # darkblue-boat
             [128, 0, 128],  # purple-bottle
             [128, 128, 0],  # greenishblue-bus
             [128, 128, 128],  # grey-car
             [0, 0, 64],  # darkmaroon-cat
             [0, 0, 192],  # red-chair
             [0, 128, 64],  # darkgreen-cow
             [0, 128, 192],  # yellowishgolden-dining table
             [128, 0, 64],  # violet-dog
             [128, 0, 192],  # pink -horse
             [128, 128, 64],  # bluishgreen-motobike
             [128, 128, 192],  # orangishpink-person,
             [0, 64, 0],  # darkgreen-plant
             [0, 64, 128],  # brown-sheep
             [0, 192, 0],  # lightgreen-sofa
             [0, 192, 128],  # lightgreen-train
             [128, 64, 0]]  # darkblue-tv/monitor
    img_onehot = []
    for colors in color:
        equality = np.equal(image, colors)
        class_map = np.all(equality, axis=-1)
        img_onehot.append(class_map)
    one_hot_encoded = np.stack(img_onehot, axis=-1)
    return  one_hot_encoded

#------------------------------------------------------------------------------
def colour_code_segmentation(image):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]

    label_values=[[0,0,0],
                 [0,0,128],
                 [0,128,0],
                 [0,128,128],
                 [128,0,0],
                 [128,0,128],
                 [128,128,0],
                 [128,128,128],
                 [0,0,64],
                 [0,0,192],
                 [0,128,64],
                 [0,128,192],
                 [128,0,64],
                 [128,0,192],
                 [128,128,64],
                 [128,128,192],
                 [0,64,0],
                 [0,64,128],
                 [0,192,0],
                 [0,192,128],
                 [128,64,0]]

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

#----------------------------------------------------------------------------------

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x
#------------------------------------------------------------------------------

def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


#-------------------------------------------------------------------------------

def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)
#-------------------------------------------------------------------------------
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies
#-------------------------------------------------------------------------------

def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou
#------------------------------------------------------------------------------

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, labels, restore=False):
        self.session = session
        self.writer = writer
        self.labels = labels

        sess = session
        ph_name = sample_name+'_mAP_ph'
        sum_name = sample_name+'_mAP'


        if restore:
            self.mAP_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.mAP_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
        else:
            self.mAP_placeholder = tf.placeholder(tf.float32, name=ph_name)
            self.mAP_summary_op = tf.summary.scalar(sum_name,
                                                    self.mAP_placeholder)

        self.placeholders = {}
        self.summary_ops = {}

        for label in labels:
            sum_name = sample_name+'_AP_'+label
            ph_name = sample_name+'_AP_ph_'+label
            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)
            self.placeholders[label] = placeholder
            self.summary_ops[label] = summary_op

    #---------------------------------------------------------------------------
    def push(self, epoch, mAP, APs):
        if not APs: return

        feed = {self.mAP_placeholder: mAP}
        tensors = [self.mAP_summary_op]
        for label in self.labels:
            feed[self.placeholders[label]] = APs[label]
            tensors.append(self.summary_ops[label])

        summaries = self.session.run(tensors, feed_dict=feed)

        for summary in summaries:
            self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, colors, restore=False):
        self.session = session
        self.writer = writer
        self.colors = colors

        sess = session
        sum_name = sample_name+'_img'
        ph_name = sample_name+'_img_ph'

        if restore:
            self.img_placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
            self.img_summary_op = sess.graph.get_tensor_by_name(sum_name+':0')

        else:
            self.img_placeholder = tf.placeholder(tf.float32, name=ph_name,
                                                  shape=[None, None, None, 3])
            self.img_summary_op = tf.summary.image(sum_name,
                                                   self.img_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):
        imgs = np.zeros((3, 512, 512, 3))
        for i, sample in enumerate(samples):
            img = cv2.resize(sample[0], (512, 512))
            for _, box in sample[1]:
                draw_box(img, box, self.colors[box.label])
            img[img>255] = 255
            img[img<0] = 0
            imgs[i] = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary_Segmentation:
    # ---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, colors, restore=False):
        self.session = session
        self.writer = writer
        self.colors = colors

        sess = session
        sum_name = sample_name + '_img'
        ph_name = sample_name + '_img_ph'

        if restore:
            self.img_placeholder = sess.graph.get_tensor_by_name(ph_name + ':0')
            self.img_summary_op = sess.graph.get_tensor_by_name(sum_name + ':0')

        else:
            self.img_placeholder = tf.placeholder(tf.float32, name=ph_name,
                                                  shape=[None, None, None, 3])
            self.img_summary_op = tf.summary.image(sum_name,
                                                   self.img_placeholder)

    # ---------------------------------------------------------------------------
    def push(self, epoch, samples):
        #imgs = np.zeros((3, 512, 512, 3))
        for sample in samples:
            sample = cv2.cvtColor(np.uint8(sample), cv2.COLOR_RGB2BGR)

        feed = {self.img_placeholder: sample}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)


#-------------------------------------------------------------------------------
class LossSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, num_samples,
                 restore=False):
        self.session = session
        self.writer = writer
        self.num_samples = num_samples
        self.loss_names = ['total', 'localization', 'confidence', 'l2','segmentation']
        self.loss_values = {}
        self.placeholders = {}

        sess = session

        summary_ops = []
        for loss in self.loss_names:
            sum_name = sample_name+'_'+loss+'_loss'
            ph_name = sample_name+'_'+loss+'_loss_ph'

            if restore:
                placeholder = sess.graph.get_tensor_by_name(ph_name+':0')
                summary_op = sess.graph.get_tensor_by_name(sum_name+':0')
            else:
                placeholder = tf.placeholder(tf.float32, name=ph_name)
                summary_op = tf.summary.scalar(sum_name, placeholder)

            self.loss_values[loss] = float(0)
            self.placeholders[loss] = placeholder
            summary_ops.append(summary_op)

        self.summary_ops = tf.summary.merge(summary_ops)

    #---------------------------------------------------------------------------
    def add(self, values, num_samples):
        for loss in self.loss_names:
            self.loss_values[loss] += values[loss]*num_samples

    #---------------------------------------------------------------------------
    def push(self, epoch):
        feed = {}
        for loss in self.loss_names:
            feed[self.placeholders[loss]] = \
                self.loss_values[loss]/self.num_samples

        summary = self.session.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        for loss in self.loss_names:
            self.loss_values[loss] = float(0)




