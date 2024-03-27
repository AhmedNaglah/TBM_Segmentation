import os
import cv2
import numpy as np


PATH = 'D:/fig_he2fibrosis/'

fnames = os.listdir(PATH)

if not os.path.exists(f'{PATH}out/'):
    os.mkdir(f'{PATH}out/')

""" for i in range(len(fnames)):
    try:
        im = cv2.imread(f'{PATH}{fnames[i]}',cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5), np.uint8)
        im = np.array(((im > 128) + 0) * 255, dtype='uint8')

        for _ in range(1):
            im = cv2.erode(im,kernel)
            im = cv2.dilate(im, kernel)

        fname_ = fnames[i].replace('.jpg', '_output.jpg')
        cv2.imwrite(f'{PATH}out/{fname_}', im)
    except:
        print('not image') """

roi_hue = [170, 260]
roi_hue_255 = [int(round(k*255/360,0)) for k in roi_hue]
roi_hue_255 = [110, 130]

for i in range(len(fnames)):
    try:
        image = fnames[i]
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
        fname_ = fnames[i].replace('.jpg', '_output.jpg')
        cv2.imwrite(f'{PATH}out/{fname_}', blue_gt)
    except:
        print(f'Corresponding image not found {image}')