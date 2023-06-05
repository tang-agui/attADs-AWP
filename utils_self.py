
from PIL import Image
import PIL.Image
import PIL.ImageOps
from utils import *
import utils
from darknet_v3 import Darknet
import numpy as np
import matplotlib.pyplot as plt
ANCHOR_PATH = "data/yolov3_anchors.txt"
DOTA_NAMES = "data/dota.names"


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img

def txt_len_read(txtfile_list):
    # for instances calculate
    len_txt = 0
    len_ins_account = []
    for txtfile_label in os.listdir(txtfile_list):  
        txtfile = os.path.abspath(os.path.join(txtfile_list, txtfile_label)) 
        if os.path.getsize(txtfile):
            myfile = open(txtfile)
            single_len = len(myfile.readlines())
            len_txt += single_len
            len_ins_account.append(single_len)

    return len_txt #, len_ins_account


from torchvision import transforms
from PIL import Image
import torch


def img_transfer(img):
    
    if isinstance(img, Image.Image):
      
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    
        img = img.view(height, width, 3).transpose(
            0, 1).transpose(0, 2).contiguous()  
        img = img.view(3, height, width)
        img = img.float().div(255.0)  # 
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(
            2, 0, 1)).float().div(255.0)
    else:
        print("unknown image type")
        exit(-1)
    return img

def bboxes_decode(YOLOoutputs, thresh = 0.4):

    anchors = utils.get_anchors(ANCHOR_PATH)
    num_anchors = len(anchors)
    class_names = utils.load_class_names(DOTA_NAMES)
    num_classes = len(class_names)

    output_single_dim = []
    for i, output in enumerate(YOLOoutputs):
        batch = output.size(0)

        h = output.size(2)
        w = output.size(3)  # 32 x 32
        output = output.view(batch, 3, 5 + num_classes, h * w)
       
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, 5 + num_classes, 3 * h * w)  # [batch, 20, x]
        output_single_dim.append(output)

    output_cat = torch.cat(output_single_dim, 2)
    # output_sigmoid = torch.sigmoid()
    output_sigmoid = torch.sigmoid(output_cat[:, 4:, :]) 

    output_sigmoid = output_sigmoid.view(16,-1) 
    output_sigmoid_obj = output_sigmoid[0,:]
    
    index_find = torch.nonzero((output_sigmoid_obj>thresh))
    index_find = index_find.view(len(index_find))
    
    output_sigmoid = output_sigmoid[:,index_find].transpose(0,1).contiguous()
    
    return output_sigmoid


