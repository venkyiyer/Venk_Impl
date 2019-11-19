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

label_defs = [
    Label("car", (142,0,0)),
    Label("person",(60,20,220)),
    Label("cyclist", (32,11,119)),
    Label("bus", (255,0,0)),
    Label("truck", (0,255,255))]

print(label_defs)

class KittiSource:
    
    def __init__(self):
        self.num_classes = len(label_defs)
        self.colors = {l.name: l.color for l in label_defs}
        self.lid2name = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []
    
    def _build_annotation_list(self, root, dataset_type):
        annot_root = root + 'Annotations/'
        annot_files = []
        with open(root + '/' + dataset_type +'.txt') as f:
            for line in f:
                annot_file = annot_root + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files.append(annot_file)
        print(len(annot_files))    
        return annot_files
    
    
    def _build_sample_list(self, root, annot_files):
        image_root = root + 'training/image_2/'
        image_seg_root = root + 'training/semantic_rgb/'
        samples = []
        
        for fn in tqdm(annot_files,unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root+doc.xpath('/annotation/filename')[0].text
                with open(fn, 'r') as f1:
                    doc1 =lxml.etree.parse(f1)
                    filename1 = image_seg_root + doc1.xpath('/annotation/filename')[0].text
                
                

                #---------------------------------------------------------------
                # Get the file dimensions
                #---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img     = cv2.imread(filename)
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
                sample = Sample(filename, boxes, imgsize, filename1)
                samples.append(sample)
                    
        return samples
    
    def _build_annotation_val_list(self, root, dataset_type):
        annot_root_valid = root + 'Annotations_valid/'
        annot_files_valid= []
        with open(root + '/' + dataset_type +'.txt') as f:
            for line in f:
                annot_file = annot_root_valid + line.strip() + '.xml'
                if os.path.exists(annot_file):
                    annot_files_valid.append(annot_file)

        return annot_files_valid
    
    
    def _build_sample_val_list(self, root, annot_files):
        image_root = root + 'valid/image_2/'
        image_seg_root = root + 'valid/semantic_rgb/'
        samples = []
        
        for fn in tqdm(annot_files,unit='samples'):
            with open(fn, 'r') as f:
                doc = lxml.etree.parse(f)
                filename = image_root+doc.xpath('/annotation/filename')[0].text
                with open(fn, 'r') as f1:
                    doc1 =lxml.etree.parse(f1)
                    filename1 = image_seg_root + doc1.xpath('/annotation/filename')[0].text
                
                

                #---------------------------------------------------------------
                # Get the file dimensions
                #---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img     = cv2.imread(filename)
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
                sample = Sample(filename, boxes, imgsize, filename1)
                samples.append(sample)
                    
        return samples
    

    
    
    def load_trainval_data(self, data_dir, valid_fraction):
        train_annot = []
        train_samples = []
        valid_annot =[]
        valid_samples =[]
        for Kitti_id in ['KittiData/']:
            root = Kitti_id
            annot = self._build_annotation_list(root,'train')
            annot_valid = self._build_annotation_val_list(root, 'val')
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
        
        
    
    def load_test_data(self, data_dir):
        root = data_dir + '/testing'
        self.test_samples = self.test_samples  = self.__build_sample_list(root, annot)
        
        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in'+data_dir)
        
        self.num_test = len(self.test_samples)

#------------------------------------------------------------------------------------------------------------

def get_source():
    return KittiSource()

        
                    
                    

        
