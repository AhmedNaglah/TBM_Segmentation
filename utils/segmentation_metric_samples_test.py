import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Naglah Deep Learning - Utility')
parser.add_argument("--dataroot", required= True, help="input directory")
parser.add_argument("--outputroot", required= True, help="input directory")

params = parser.parse_args()
PATH = params.dataroot
OUTPATH = params.outputroot

roi_hue = [170, 260]
roi_hue_255 = [int(round(k*255/360,0)) for k in roi_hue]
roi_hue_255 = [110, 130]

images = os.listdir(PATH)
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

for image in images:
    if image.endswith('real_B.png'):
        try:
            print(os.path.join(PATH, image))
            im = cv2.imread(os.path.join(PATH, image),cv2.IMREAD_COLOR)
            im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            low_blue = (roi_hue_255[0],50,50)
            high_blue = (roi_hue_255[1],255,255)

            blue_mask = cv2.inRange(im_hsv, low_blue, high_blue)
            blue_gt = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)

            image_input = image.replace('real_B.png', 'fake_B.png')
            im_input = cv2.imread(os.path.join(PATH, image_input),cv2.IMREAD_COLOR)

            out = cv2.hconcat((im_input, blue_gt))

            cv2.imwrite(os.path.join(OUTPATH, image), out)
        except:
            print(f'Corresponding image not found {image}')