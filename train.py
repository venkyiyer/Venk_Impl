#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   07.09.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import os
import cv2
import multiprocessing as mp
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt


from average_precision import APCalculator, APs2mAP
from training_data import TrainingData
from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdvgg import SSDVGG
from utils import *
from tqdm import tqdm
from source_VOC import  Colour_code_segmentation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

#-------------------------------------------------------------------------------
def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--name', default='test-3000Dec',
                        help='project name')
    parser.add_argument('--data-dir', default='VOC',
                        help='data directory')
    parser.add_argument('--vgg-dir', default='vgg_graph',
                        help='directory for the VGG-16 model')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--tensorboard-dir', default="tb-3000Dec",
                        help='name of the tensorboard data directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='checkpoint interval')
    parser.add_argument('--lr-values', type=str, default='0.00001; 0.00001;0.00001',
                        help='learning rate values')
    parser.add_argument('--lr-boundaries', type=str, default='320000;400000',
                        help='learning rate chage boundaries (in batches)')
    parser.add_argument('--momentum', type=float, default=0.0009,
                        help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='L2 normalization factor')
    parser.add_argument('--continue-training', type=str2bool, default='False',
                        help='continue training from the latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=6,
                        help='number of parallel generators')

    args = parser.parse_args()

    print('[i] Project name:         ', args.name)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] VGG directory:        ', args.vgg_dir)
    print('[i] # epochs:             ', args.epochs)
    print('[i] Batch size:           ', args.batch_size)
    print('[i] Tensorboard directory:', args.tensorboard_dir)
    print('[i] Checkpoint interval:  ', args.checkpoint_interval)
    print('[i] Learning rate values: ', args.lr_values)
    print('[i] Learning rate boundaries: ', args.lr_boundaries)
    print('[i] Momentum:             ', args.momentum)
    print('[i] Weight decay:         ', args.weight_decay)
    print('[i] Continue:             ', args.continue_training)
    print('[i] Number of workers:    ', args.num_workers)

    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    start_epoch = 0
    if args.continue_training:
        state = tf.train.get_checkpoint_state(args.name)
        if state is None:
            print('[!] No network state found in ' + args.name)
            return 1

        ckpt_paths = state.all_model_checkpoint_paths
        if not ckpt_paths:
            print('[!] No network state found in ' + args.name)
            return 1

        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph', metagraph_file)
            return 1
        start_epoch = last_epoch

    #---------------------------------------------------------------------------
    # Create a project directory
    #---------------------------------------------------------------------------
    else:
        try:
            if not os.path.exists(args.name):
                print('[i] Creating directory {}...'.format(args.name))
                os.makedirs(args.name)
        except (IOError) as e:
            print('[!]', str(e))
            return 1

    print('[i] Starting at epoch:    ', start_epoch+1)

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        td = TrainingData(args.data_dir)
        print('[i] # training samples:   ', td.num_train)
        print('[i] # validation samples: ', td.num_valid)
        print('[i] # classes:            ', td.num_classes)
        print('[i] Image size:           ', td.preset.image_size)
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1
        
       
    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(td.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(td.num_valid/args.batch_size))

        global_step = None
        if start_epoch == 0:
            lr_values = args.lr_values.split(';')
            try:
                lr_values = [float(x) for x in lr_values]
            except ValueError:
                print('[!] Learning rate values must be floats')
                sys.exit(1)

            lr_boundaries = args.lr_boundaries.split(';')
            try:
                lr_boundaries = [int(x) for x in lr_boundaries]
            except ValueError:
                print('[!] Learning rate boundaries must be ints')
                sys.exit(1)

            ret = compute_lr(lr_values, lr_boundaries)
            learning_rate, global_step = ret

        net = SSDVGG(sess, td.preset)
        if start_epoch != 0:
            net.build_from_metagraph(metagraph_file, checkpoint_file)
            net.build_optimizer_from_metagraph()
        else:
            net.build_from_vgg(args.vgg_dir, td.num_classes, a_trous=False)
            net.build_optimizer(learning_rate=learning_rate,
                                global_step=global_step,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

        initialize_uninitialized_variables(sess)


        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        summary_writer = tf.summary.FileWriter(args.tensorboard_dir,
                                               sess.graph)
        saver = tf.train.Saver(max_to_keep=20)

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        avg_loss_per_epoch = []
        avg_scores_per_epoch = []
        avg_iou_per_epoch = []

        anchors = get_anchors_for_preset(td.preset)
        training_ap_calc = APCalculator()
        validation_ap_calc = APCalculator()

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        training_ap = PrecisionSummary(sess, summary_writer, 'training',
                                       td.lname2id.keys(), restore)
        validation_ap = PrecisionSummary(sess, summary_writer, 'validation',
                                         td.lname2id.keys(), restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training',
                                     td.label_colors, restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation',
                                       td.label_colors, restore)
        training_loss = LossSummary(sess, summary_writer, 'training',
                                    td.num_train, restore)
        validation_loss = LossSummary(sess, summary_writer, 'validation',
                                      td.num_valid, restore)

        #-----------------------------------------------------------------------
        # Get the initial snapshot of the network
        #-----------------------------------------------------------------------
        net_summary_ops = net.build_summaries(restore)
        if start_epoch == 0:
            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, 0)
        summary_writer.flush()

        #-----------------------------------------------------------------------
        # Cycle through the epoch
        #-----------------------------------------------------------------------
        print('[i] Training...')
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []
            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            generator = td.train_generator(args.batch_size, args.num_workers)
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for x, y, gt_boxes, img_seg_gt, imgseg_gt_to_compare in tqdm(generator, total=n_train_batches,
                                       desc=description, unit='batches'):

                cv2.imwrite('img_seg_to_compare_training.png', np.squeeze(imgseg_gt_to_compare))
                rev = np.squeeze(img_seg_gt)
                gt_rev_onehot = reverse_one_hot(rev)
                output_seg_output = colour_code_segmentation(gt_rev_onehot)
                cv2.imwrite('revereceonehot.png', output_seg_output)
                print('img_seg_gt',img_seg_gt)
                exit()
                if len(training_imgs_samples) < 3:
                    saved_images = np.copy(x[:3])

                feed = {net.image_input: x, net.labels: y, net.label_seg_gt:img_seg_gt} #
                fcn32, result,loss_batch, _ = sess.run([net.fcn32_upsampled,net.result, net.losses,net.optimizer], feed_dict=feed)


                if math.isnan(loss_batch['total']):
                    print('[!] total loss is NaN.')

                training_loss.add(loss_batch, x.shape[0])

                if e == 0: continue

                for i in range(result.shape[0]):
                   boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                   boxes = suppress_overlaps(boxes)
                   training_ap_calc.add_detections(gt_boxes[i], boxes)

                   if len(training_imgs_samples) < 3:
                      training_imgs_samples.append((saved_images[i], boxes))

            #-------------------------------------------------------------------
            # Validate
            #-------------------------------------------------------------------
            generator = td.valid_generator(args.batch_size, args.num_workers)
            description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)
            counter = 0
            if not os.path.isdir("%s/%04d" % ("checkpoints3Dec", e)):
                os.makedirs("%s/%04d" % ("checkpoints3Dec", e))

            target = open("%s/%04d/val_scores.csv" % ("checkpoints3Dec", e), 'w')
            target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % str(counter))


            for x, y, gt_boxes, img_seg_gt, imgseg_gt_to_compare in tqdm(generator, total=n_valid_batches,
                                       desc=description, unit='batches'):


                gt_rev_onehot = reverse_one_hot(one_hot_encode(np.squeeze(imgseg_gt_to_compare)))

                feed = {net.image_input: x, net.labels: y, net.label_seg_gt:img_seg_gt} #,
                result,output_seg,loss_batch = sess.run([net.result,net.logits_seg, net.losses], feed_dict=feed)


                output_image = np.array(output_seg[0,:,:,:])
                output_seg_rev = reverse_one_hot(output_image)
                output_seg_output = colour_code_segmentation(output_seg_rev)

                accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred=output_seg_rev,
                                                                                             label=gt_rev_onehot,
                                                                                             num_classes=21)

                filename = str(counter)
                target.write("%s, %f, %f, %f, %f, %f" % (filename, accuracy, prec, rec, f1, iou))
                for item in class_accuracies:
                    target.write(", %f" % (item))
                target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)


                #file_name = os.path.splitext(file_name)[0]
                original_gt = colour_code_segmentation(gt_rev_onehot)

                cv2.imwrite("%s/%04d/%s_gt.png" % ("checkpoints3Dec", e, filename), original_gt)
                cv2.imwrite("%s/%04d/%s_pred.png" % ("checkpoints3Dec", e, filename),output_seg_output)


                counter += 1
                if e == 0: continue


                for i in range(result.shape[0]):
                   boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                   boxes = suppress_overlaps(boxes)
                   validation_ap_calc.add_detections(gt_boxes[i], boxes)

                   if len(validation_imgs_samples) < 3:
                      validation_imgs_samples.append((np.copy(x[i]), boxes))

            target.close()
            #-------------------------------------------------------------------
            # Write summaries
            #-------------------------------------------------------------------
            avg_score = np.mean(scores_list)
            # class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)
            avg_iou_per_epoch.append(avg_iou)

            print("\nAverage validation accuracy for epoch # %04d = %f" % (e, avg_score))
            #print("Average per class validation accuracies for epoch # %04d:" % (e))
            # for index, item in enumerate(class_avg_scores):
            #     print("%s = %f" % (class_names_list[index], item))
            print("Validation precision = ", avg_precision)
            print("Validation recall = ", avg_recall)
            print("Validation F1 score = ", avg_f1)
            print("Validation IoU score = ", avg_iou)

            #print('current_loss = %.4f ',loss_batch)
            training_loss.push(e+1)
            validation_loss.push(e+1)

            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, e+1)

            APs = training_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            training_ap.push(e+1, mAP, APs)

            APs = validation_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            validation_ap.push(e+1, mAP, APs)

            training_ap_calc.clear()
            validation_ap_calc.clear()

            training_imgs.push(e+1, training_imgs_samples)
            validation_imgs.push(e+1, validation_imgs_samples)
            #validation_imgs_seg.push(e+1, output_seg_output)

            summary_writer.flush()

            #-------------------------------------------------------------------
            # Save a checktpoint
            #-------------------------------------------------------------------
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)

        checkpoint = '{}/final.ckpt'.format(args.name)
        saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:', checkpoint)

    return 0

if __name__ == '__main__':
    sys.exit(main())
