import lxml.etree
import random
import math
import cv2
import os

import numpy as np

from utils import Label, Box, Sample, Size
from utils import abs2prop
from glob import glob
from tqdm import tqdm
# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
label_defs = [
    Label('aeroplane', (0, 0, 0)), # RGB - 128,0,0
    Label('bicycle', (111, 74, 0)), #0,128,0
    Label('bird', (81, 0, 81)), #128,128,0
    Label('boat', (128, 64, 128)), #0,0,128
    Label('bottle', (244, 35, 232)), #128,0,128
    Label('bus', (230, 150, 140)), #0,128,128
    Label('car', (70, 70, 70)), #128,128,128
    Label('cat', (102, 102, 156)), #64,0,0
    Label('chair', (190, 153, 153)), # 192,0,0
    Label('cow', (150, 120, 90)), #64,128,0
    Label('diningtable', (153, 153, 153)), #192,128,0
    Label('dog', (250, 170, 30)), #64,0,128
    Label('horse', (220, 220, 0)), #192,0,128
    Label('motorbike', (107, 142, 35)), #64,128,128
    Label('person', (52, 151, 52)), # RGB - 192,128,128
    Label('pottedplant', (70, 130, 180)), #0,64,0
    Label('sheep', (220, 20, 60)), #128,64,0
    Label('sofa', (0, 0, 142)), #0,192,0
    Label('train', (0, 0, 230)), # 128,192,0
    Label('tvmonitor', (119, 11, 32))]#RGB -0,64,128

Colour_code_segmentation = [[0, 0, 0],  # black-background,
             [0, 0, 128],  # maroon-aeroplane,
             [0, 128, 0],  # darkgreen-bicycle
             [0, 128, 128],  # gold-bird,
             [128, 0, 0],  # darkblue-boat,
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
             [128, 64, 0]]  # darkblue-tv/monitor]

class KittiSource:
    
    def __init__(self):
        self.num_classes = len(label_defs)
        self.colors = {l.name: l.color for l in label_defs}
        self.colors_seg = Colour_code_segmentation
        self.colors_seg_classes = len(Colour_code_segmentation)
        self.lid2name = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []

    def _build_annotation_list(self, root, dataset_type):
        annot_root = root + 'VOC_Annotations/'
        annot_files = []
        with open(root + '/' + dataset_type +'.txt') as f:
            for line in f:
                annot_file = annot_root + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files.append(annot_file)

        return annot_files
    
    
    def _build_sample_list(self, root, annot_files):
        image_root = root + 'VOC_Jpeg/'
        image_seg_root = root + 'VOC_Segmentation/'
        samples = []
        
        for fn in tqdm(annot_files,unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root+doc.xpath('/annotation/filename')[0].text
                with open(fn, 'r') as f1:
                    doc1 =lxml.etree.parse(f1)
                    seg_gt = image_seg_root + doc1.xpath('/annotation/filename')[0].text
                    seg_gt = seg_gt.replace('jpg','png')
                    seg_gt_to_compare = seg_gt

                #---------------------------------------------------------------
                # Get the file dimensions
                #---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img     = cv2.imread(filename)
                img_seg_gt = cv2.imread(seg_gt)
                imgsize = Size(img.shape[1], img.shape[0])
                

                #---------------------------------------------------------------
                # Get boxes for all the objects
                #---------------------------------------------------------------
                boxes    = []
                objects  = doc.xpath('/annotation/object')
                for obj in objects:
                    #-----------------------------------------------------------
                    # Get the properties of the box and convert them to the
                    # proportional terms
                    #-----------------------------------------------------------
                    label = obj.xpath('name')[0].text
                    xmin  = int(float(obj.xpath('bndbox/xmin')[0].text))
                    xmax  = int(float(obj.xpath('bndbox/xmax')[0].text))
                    ymin  = int(float(obj.xpath('bndbox/ymin')[0].text))
                    ymax  = int(float(obj.xpath('bndbox/ymax')[0].text))
                    center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                    box = Box(label, self.lname2id[label], center, size)
                    boxes.append(box)
                if not boxes:
                    continue
                sample = Sample(filename, boxes, imgsize, seg_gt, seg_gt_to_compare)
                samples.append(sample)
                    
        return samples
    
    def _build_annotation_val_list(self, root, dataset_type):
        annot_root_valid = root + 'VOC_Annotations/'
        annot_files_valid= []
        with open(root + '/' + dataset_type +'.txt') as f:
            for line in f:
                annot_file = annot_root_valid + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files_valid.append(annot_file)

        return annot_files_valid
    
    
    def _build_sample_val_list(self, root, annot_files):
        image_root = root + 'VOC_Jpeg/'
        image_seg_root = root + 'VOC_Segmentation/'
        samples = []
        
        for fn in tqdm(annot_files,unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root+doc.xpath('/annotation/filename')[0].text
                with open(fn, 'r') as f1:
                    doc1 =lxml.etree.parse(f1)
                    seg_gt = image_seg_root + doc1.xpath('/annotation/filename')[0].text
                    seg_gt = seg_gt.replace('jpg', 'png')
                    seg_gt_to_compare = seg_gt

                #---------------------------------------------------------------
                # Get the file dimensions
                #---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img     = cv2.imread(filename)
                img_seg_gt = cv2.imread(seg_gt)
                imgsize = Size(img.shape[1], img.shape[0])
                

                #---------------------------------------------------------------
                # Get boxes for all the objects
                #---------------------------------------------------------------
                boxes    = []
                objects  = doc.xpath('/annotation/object')
                for obj in objects:
                    #-----------------------------------------------------------
                    # Get the properties of the box and convert them to the
                    # proportional terms
                    #-----------------------------------------------------------
                    label = obj.xpath('name')[0].text
                    xmin  = int(float(obj.xpath('bndbox/xmin')[0].text))
                    xmax  = int(float(obj.xpath('bndbox/xmax')[0].text))
                    ymin  = int(float(obj.xpath('bndbox/ymin')[0].text))
                    ymax  = int(float(obj.xpath('bndbox/ymax')[0].text))
                    center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                    box = Box(label, self.lname2id[label], center, size)
                    boxes.append(box)
                if not boxes:
                    continue
                sample = Sample(filename, boxes, imgsize, seg_gt, seg_gt_to_compare)
                samples.append(sample)
                    
        return samples

    def load_trainval_data(self, data_dir, valid_fraction):
        train_annot = []
        train_samples = []
        valid_annot =[]
        valid_samples =[]
        for voc_id in ['VOC/']:
            root = voc_id
            annot = self._build_annotation_list(root,'train_2750') # Exp_train  training_2330
            annot_valid = self._build_annotation_val_list(root, 'valid_146') # Exp_valid
            train_annot += annot
            valid_annot += annot_valid
            train_samples += self._build_sample_list(root, annot)
            valid_samples += self._build_sample_val_list(root, annot_valid)            

        self.valid_samples = valid_samples
        self.train_samples = train_samples
       
        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)

        if valid_fraction > 0:
            if len(self.valid_samples) == 0:
                raise RuntimeError('No validation samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)
        
    def __build_annotation_test_list(self, root, dataset_type):
        annot_root_test = root + 'VOC_Annotations/'
        annot_files_test = []
        with open(root + '/' + dataset_type + '.txt') as f:
            for line in f:
                annot_file = annot_root_test + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files_test.append(annot_file)

        return annot_files_test

    def __build_sample_test_list(self, root, annot_test):
        image_root = root + 'VOC_Jpeg/'
        image_seg_root = root + 'VOC_Segmentation/'
        samples = []

        for fn in tqdm(annot_test, unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root + doc.xpath('/annotation/filename')[0].text
                with open(fn, 'r') as f1:
                    doc1 = lxml.etree.parse(f1)
                    seg_gt = image_seg_root + doc1.xpath('/annotation/filename')[0].text
                    seg_gt = seg_gt.replace('jpg', 'png')
                    seg_gt_to_compare = seg_gt
                # ---------------------------------------------------------------
                # Get the file dimensions
                # ---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img = cv2.imread(filename)
                img_seg_gt = cv2.imread(seg_gt)
                imgsize = Size(img.shape[1], img.shape[0])

                # ---------------------------------------------------------------
                # Get boxes for all the objects
                # ---------------------------------------------------------------
                boxes = []
                objects = doc.xpath('/annotation/object')
                for obj in objects:
                    # -----------------------------------------------------------
                    # Get the properties of the box and convert them to the
                    # proportional terms
                    # -----------------------------------------------------------
                    label = obj.xpath('name')[0].text
                    xmin = int(float(obj.xpath('bndbox/xmin')[0].text))
                    xmax = int(float(obj.xpath('bndbox/xmax')[0].text))
                    ymin = int(float(obj.xpath('bndbox/ymin')[0].text))
                    ymax = int(float(obj.xpath('bndbox/ymax')[0].text))
                    center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                    box = Box(label, self.lname2id[label], center, size)
                    boxes.append(box)
                if not boxes:
                    continue
                sample = Sample(filename, boxes, imgsize, seg_gt, seg_gt_to_compare)
                samples.append(sample)

        return samples

    
    def load_test_data(self, data_dir):
        """
        Load the test data
        :param data_dir: the directory where the dataset's file are stored
        """
        root = data_dir + '/'
        annot_test = self.__build_annotation_test_list(root, 'test_17') # Exp_testing
        self.test_samples  = self.__build_sample_test_list(root, annot_test)

        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' + data_dir)

        self.num_test  = len(self.test_samples)

#------------------------------------------------------------------------------------------------------------

def get_source():
    return KittiSource()

        
                    
                    

        
