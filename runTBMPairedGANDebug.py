import os
import logging
from modules.ftu import Tubule
from modules.slide import Slide
from modules.dsa import DSAFolder, DSAItem, DSA

import cv2
import numpy as np

import argparse

import histomicstk as htk
import os
import tensorflow as tf
import numpy as np
import cv2
from models.cycleGAN512 import cycleGAN512

def GANInference(im, checkpoint_path): # Takes BGR image and Return BGR
    from models.cycleGAN512 import cycleGAN512

    def normalize(input_image):
        input_image = (input_image / 127.5) - 1

        return input_image

    def TF2CV(im):
        img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
        img_ = np.array(tf.keras.utils.array_to_img(img),dtype='uint8')
        #img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
        return img_

    print("1 | GANInference")
    cycleGAN = cycleGAN512()
    cycleGAN.built = True
    print("2")
    print(checkpoint_path)
    cycleGAN.load_weights(checkpoint_path)
    print("3")

    # Model is always trained on RGB internally but expect BGR
    im_ = normalize(im)
    he_ = np.expand_dims(im_, axis=0)
    print("4")

    mt_virtual = cycleGAN.predict(he_)
    print("5")

    mt_virtual_ = TF2CV(mt_virtual)
    print("6")

    return mt_virtual_

def segmentTBMPair(im):

    def applyThresholdEosin(patch):
        lower = 170
        upper = 255
        mask = cv2.inRange(patch, lower, upper)
        
        return np.array(255-mask, dtype='uint8')

    def smoothBinary(mask):
        kn = 2
        iterat = 2
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
        return mask

    def stainDeconv(im_):
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

        stains = ['hematoxylin',  # nuclei stain
                    'eosin',        # cytoplasm stain
                    'null']         # set to null if input contains only two stains

        W = np.array([stain_color_map[st] for st in stains]).T

        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(im_, W)

        eosinStain = imDeconvolved.Stains[:, :, 1]

        return eosinStain

    _, w, _ = np.shape(im)

    wn = w//2

    msk = im[:, wn:, :]
    im = im[:, :wn, :]

    thresholdArea = 0.01

    #imgYuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    #imgYuv[:,:,0] = cv2.equalizeHist(imgYuv[:,:,0])
    #imEquBGR = cv2.cvtColor(imgYuv, cv2.COLOR_YUV2BGR)

    #eosin = stainDeconv(imEquBGR)

    eosin = stainDeconv(im)
    #eosinEqu = cv2.equalizeHist(eosin)

    mask = applyThresholdEosin(eosin)
    maskSmooth = smoothBinary(mask)
    mask3d = cv2.cvtColor(maskSmooth, cv2.COLOR_GRAY2RGB)
    eosin3 = cv2.cvtColor(eosin, cv2.COLOR_GRAY2RGB)

    #contours, _ = cv2.findContours(maskSmooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c_max = max(contours, key = cv2.contourArea)
    area_min = cv2.contourArea(c_max) * thresholdArea

    contours_filtered = []

    amin = 100

    for contour, hie in zip(contours, hierarchy[0]):
        area = cv2.contourArea(contour)
        if area > area_min:
            if area > amin:
                msk_ = msk*0
                interThickness = 24
                cv2.drawContours(msk_, [contour], -1, [255,255,255], interThickness)
                if np.sum(msk_*msk)>0:
                    contours_filtered.append((contour,hie))

    return contours_filtered

def segmentTBMPairLUT(im):

    def applyThresholdEosin(patch):
        lower = 230
        upper = 255
        mask = cv2.inRange(patch, lower, upper)
        
        return np.array(255-mask, dtype='uint8')

    def smoothBinary(mask):
        kn = 2
        iterat = 2
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
        return mask

    def stainDeconv(im_):
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

        stains = ['hematoxylin',  # nuclei stain
                    'eosin',        # cytoplasm stain
                    'null']         # set to null if input contains only two stains

        W = np.array([stain_color_map[st] for st in stains]).T

        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(im_, W)

        eosinStain = imDeconvolved.Stains[:, :, 1]

        return eosinStain

    _, w, _ = np.shape(im)

    wn = w//2

    msk = im[:, wn:, :]
    im = im[:, :wn, :]

    thresholdArea = 0.0001

    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')

    eosin = stainDeconv(im)
    eosinEqu = cv2.LUT(eosin, table)

    mask = applyThresholdEosin(eosinEqu)
    maskSmooth = smoothBinary(mask)
    mask3d = cv2.cvtColor(maskSmooth, cv2.COLOR_GRAY2RGB)
    eosin3 = cv2.cvtColor(eosin, cv2.COLOR_GRAY2RGB)

    #contours, _ = cv2.findContours(maskSmooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c_max = max(contours, key = cv2.contourArea)
    area_min = cv2.contourArea(c_max) * thresholdArea

    contours_filtered = []

    amin = 100

    for contour, hie in zip(contours, hierarchy[0]):
        area = cv2.contourArea(contour)
        if area > area_min:
            if area > amin:
                msk_ = msk*0
                interThickness = 2
                cv2.drawContours(msk_, [contour], -1, [255,255,255], interThickness)
                if np.sum(msk_*msk)>0:
                    contours_filtered.append((contour,hie))

    return contours_filtered

def generateAnno(contoursColors):

    annos = []
    elems = []

    for q in range(len(contoursColors)):
        cnts = contoursColors[q]['cnt']
        color = contoursColors[q]['color']

        for f in range(len(cnts)):
            cnt = cnts[f][0]
            hie = cnts[f][1]
            print(hie)
            if hie[3]==-1:
                points = []
                for i in cnt:
                    for j in i:
                        x = int(j[0]) + contoursColors[q]['x'] 
                        y = int(j[1]) + contoursColors[q]['y'] 
                        points.append([x, y, 0])

                elem = {
                    "type": "polyline",
                    "lineColor": f"rgb({color})", 
                    "lineWidth": 3, 
                    "fillColor": f"rgba({color}, 0.5)",
                    "points": None,
                    "closed": True
                }

                elem["points"] = points

                if len(points)>1:
                    if (f+1)<len(cnts) and cnts[f+1][1][3]==-1: #No Children
                        pass
                    else: # Get Children
                        z = f+1
                        holes = []
                        while True:
                            hole = []
                            try:
                                cntChild = cnts[z][0]
                                hieChild = cnts[z][1]
                            except:
                                break
                            if hieChild[3]==-1:
                                break
                            for i in cntChild:
                                for j in i:
                                    x = int(j[0]) + contoursColors[q]['x'] 
                                    y = int(j[1]) + contoursColors[q]['y'] 
                                    hole.append([x, y, 0])
                            holes.append(hole)
                            z+=1
                        elem["holes"] = holes
                    elems.append(elem)

    anno = {
        "name": 'tbmPairedLUT', 
        "description": 'tbmPairedLUT',  
        "elements": None                        
    }
    anno["elements"] = elems
    annos.append(anno)

    return annos

def applyThreshold(patch):
    
    try:
        hue_min = int(295 * 180/360)
        hue_max = int(315 * 180/360)
        val_min = 0
        val_max = 255
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lower_white = np.array([hue_min,0,val_min], dtype=np.uint8)
        upper_white = np.array([hue_max,255,val_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask  
    except:
        print("Except inside apply threshold")  


def main():

    config = {
        "svsBase": "/blue/pinaki.sarder/nlucarelli/kpmp_new/",
        "fid": "65fc4d4ed2f45e99a916b24c",
        "outputdir": "/orange/pinaki.sarder/ahmed.naglah/data/GANGAN3",
        "username": "ahmednaglah",
        "password": "Netzwork_171819",
        "apiUrl": "https://athena.rc.ufl.edu/api/v1",
        "patchSize": 512 ,
        "layerName": "tubules" ,
        "name": "GANSeg222" ,
        "checkpoint_path": "/orange/pinaki.sarder/ahmed.naglah/data/kpmpCycleGAN/output_kpmpCycleGAN3/training_checkpoints/ckpt-1"
    }

    outdir = config['outputdir']

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dsa = DSA(config)

    dsaFolder = DSAFolder(config)
    items = dsaFolder.items

    for i in range(len(items)):
        fid, svsname = items[i]

        try:
            item = DSAItem(config, fid, svsname)
            anno = item.annotations

            if len(anno)>0:
                slide = Slide(config, svsname)

                slide.extractForeground()

                slide.createBinary(anno)

                patches = slide.extractPairs(config['patchSize'])

                print(len(patches))

                contoursColors = []
                print(f"Processing patches starting ... ")
                for key in patches:
                    logging.warning(f"Processing patches ... {key}")
                    patch = patches[key]
                    print(f"Processing patches ... {key}")
                    #patchGAN = GANInference(patch, config['checkpoint_path'])

                    def normalize(input_image):
                        input_image = (input_image / 127.5) - 1

                        return input_image

                    def TF2CV(im):
                        img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
                        img_ = np.array(tf.keras.utils.array_to_img(img),dtype='uint8')
                        #img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                        return img_

                    print("1 | GANInference | w/o main")
                    cycleGAN = cycleGAN512()
                    cycleGAN.built = True
                    print("2")
                    print(config['checkpoint_path'])
                    cycleGAN.load_weights(config['checkpoint_path'])
                    print("3")

                    # Model is always trained on RGB internally but expect BGR
                    im_ = normalize(patch)
                    he_ = np.expand_dims(im_, axis=0)
                    print("4")

                    mt_virtual = cycleGAN(he_)
                    print("5")

                    patchGAN = TF2CV(mt_virtual)
                    print("6")

                    contours = segmentTBMPairLUT(patchGAN)
                    x, y = key.split('_')
                    contoursColors.append({'x': int(x), 'y': int(y) , 'cnt': contours, 'color': '255, 20, 20'})
                    #cv2.imwrite(f"{outdir}/{key}.png", patchGAN)
                annos = generateAnno(contoursColors)
                dsa.postAnno(fid, annos)
            else:
                logging.warning(f"No annotation found in {fid} __ {svsname}")
        except:
            logging.warning(f"Exeption in {fid} __ {svsname}")

if __name__=="__main__":
    main()

