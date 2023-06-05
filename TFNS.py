
import time
import numpy as np
import torchvision.transforms as transforms
from TDE_main_hybrid import FDE
from darknet_v3 import Darknet
import os
import utils_self
import utils
from PIL import Image
import fnmatch
import TDE_main_hybrid
import cv2 as cv2
import Automold as am
import sys
import torch


print('hyper-paramenters,population size : {}, generations : {}, mutation factor (F) : {}, crossover factor (CR) : {}'.format(
    TDE_main_hybrid.population_size, TDE_main_hybrid.generations, TDE_main_hybrid.F, TDE_main_hybrid.CR))



print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))

imgdir = 'imgspath'
clean_labdir = "labels"
savedir = "savepath"


print("savedir : ", savedir)
cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "checkpoint"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()

attacked_total_count = 0
clean_total_count = 0

img_size = model.height
img_width = model.width
class_names = utils.load_class_names('data/dota.names')

instances_clean = []
instances_after_attack = []

t_begin = time.time()


for imgfile in os.listdir(imgdir):

    t_single_begin = time.time()
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  
        print("new image") 
        name = os.path.splitext(imgfile)[0]

        txtname = name + '.txt'  
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
        print("image file path is : ", imgfile)
        img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (608, 608))

        tru_lab = utils.read_truths_pre_7(txtpath)  

        clean_total_count += len(tru_lab)  

        if len(tru_lab):  
       
            images_attack = FDE(img)
    
            images_attack = transforms.ToPILImage(
                'RGB')(images_attack.cpu())  
            save_name_attacked = name + '.png'
            attacked_dir = os.path.join(
                savedir, 'img_attacked/', save_name_attacked)
            images_attack.save(attacked_dir)

            boxes_cls_attack = utils.do_detect(
                model, images_attack, 0.01, 0.4, True)
            boxes_attack_all = []  
            boxes_attack = []
            for box in boxes_cls_attack:
                obj_conf = box[4]
                if obj_conf >=0.4:
                    boxes_attack.append(box)
                boxes_attack_all.append(box)


            pre_name = name + ".png"  
            pre_dir = os.path.join(
                savedir, 'patched_pre/', pre_name)
            utils.plot_boxes(images_attack, boxes_attack, pre_dir,
                             class_names=class_names)  
            txtpath_pre = os.path.abspath(os.path.join(
                savedir, 'yolo-labels', txtname))
            textfile_pre = open(txtpath_pre, 'w+')  

            txtpath_w_conf_pre = os.path.abspath(os.path.join(
                savedir, 'yolo-labels_w_conf', txtname))
            textfile_w_conf_pre = open(
                txtpath_w_conf_pre, 'w+')  

            for box in boxes_attack_all:
                textfile_w_conf_pre.write(
                    f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
                obj_conf = box[4]
                if obj_conf > 0.4:
                    textfile_pre.write(
                        f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
            textfile_pre.close()
            textfile_w_conf_pre.close()
            
            attacked_lab = utils.read_truths_pre_7(txtpath_pre)
            attacked_total_count += len(attacked_lab)
            
            print("single image tru-instances : ", len(tru_lab), "instances after attack : ",
                  len(attacked_lab), "instances gap : ", (len(tru_lab)-len(attacked_lab)))
        attacked_count = clean_total_count - attacked_total_count
