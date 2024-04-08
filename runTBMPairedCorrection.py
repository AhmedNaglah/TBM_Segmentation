import os
import logging
from modules.ftu import Tubule
from modules.slide import Slide
from modules.dsa import DSAFolder, DSAItem, DSA

import cv2
import numpy as np

import pandas as pd
import argparse

import histomicstk as htk


def segmentTBMPair(im, low, threshArea):

    def applyThresholdEosin(patch):
        lower = low
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

    thresholdArea = threshArea

    eosin = stainDeconv(im)

    mask = applyThresholdEosin(eosin)
    maskSmooth = smoothBinary(mask)


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
                interThickness = 1
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
        "name": 'tbmHandcraftedThre', 
        "description": 'tbmHandcraftedThre',  
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

    slides222 = {
    "S-1908-009781_PAS_2of2.svs": {"svs": "S-1908-009781_PAS_2of2.svs", "lower": 220, "thresholdArea": 0.05}, 
    "S-1908-009782_PAS_1of2.svs": {"svs": "S-1908-009782_PAS_1of2.svs", "lower": 230, "thresholdArea": 0.05}, 
    "S-2001-019412_PAS_2of2.svs": {"svs": "S-2001-019412_PAS_2of2.svs", "lower": 170, "thresholdArea": 0.05}, 
    "S-2001-019413_PAS_1of2.svs": {"svs": "S-2001-019413_PAS_1of2.svs", "lower": 185, "thresholdArea": 0.007}, 
    "S-2006-004999_PAS_2of2.svs": {"svs": "S-2006-004999_PAS_2of2.svs", "lower": 195, "thresholdArea": 0.007}, 	
    "S-2203-008879_PAS_1of2.svs": {"svs": "S-2203-008879_PAS_1of2.svs", "lower": 190, "thresholdArea": 0.008}, 	
    "S-2203-008878_PAS_2of2.svs": {"svs": "S-2203-008878_PAS_2of2.svs", "lower": 170, "thresholdArea": 0.008}, 	
    "S-2111-008283_PAS_2of2.svs": {"svs": "S-2111-008283_PAS_2of2.svs", "lower": 200, "thresholdArea": 0.005}, 	
    "S-2112-014312_PAS_1of1.svs": {"svs": "S-2112-014312_PAS_1of1.svs", "lower": 210, "thresholdArea": 0.08}, 	
    "S-2001-000042_PAS_2of2.svs": {"svs": "S-2001-000042_PAS_2of2.svs", "lower": 180, "thresholdArea": 0.012}, 	
    "S-2103-004716_PAS_2of2.svs": {"svs": "S-2103-004716_PAS_2of2.svs", "lower": 190, "thresholdArea": 0.07}, 	
    }
    parser = argparse.ArgumentParser(
                        prog='DSA FTU Annotations to Patches',
                        description='Extract DSA FTU Annotations to Patches',
                        epilog='Text at the bottom of help')

    parser.add_argument('--svsBase', help='/blue/pinaki.sarder/...')     
    parser.add_argument('--fid', help='folder id in DSA')     
    parser.add_argument('--outputdir', help='/orange/pinaki.sarder/ahmed.naglah/...')     
    parser.add_argument('--username', help='username')     
    parser.add_argument('--password', help='password')
    parser.add_argument('--patchSize', help='patchSize', type=int, default=512)
    parser.add_argument('--thresholdArea', help='thresholdArea', type=float, default=0.3)
    parser.add_argument('--spatialResolution', help='spatialResolution', type=float, default=0.25)      
    parser.add_argument('--apiUrl', help='https://athena.rc.ufl.edu/api/v1')   
    parser.add_argument('--name', help='a name for the pipeline', default='defaultNaglahPipeline')     
    parser.add_argument('--layerName', help='annotation layer on DSA', type=str)

    args = parser.parse_args()

    config = vars(args)

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
                for key in patches:
                    try:
                        patch = patches[key]
                        low = slides222[svsname]['lower']
                        thrArea = slides222[svsname]['thresholdArea']
                        contours = segmentTBMPair(patch, low, thrArea)
                        x, y = key.split('_')
                        contoursColors.append({'x': int(x), 'y': int(y) , 'cnt': contours, 'color': '255, 20, 20'})
                        cv2.imwrite(f"{outdir}/{key}.png", patch)
                    except:
                        pass
                annos = generateAnno(contoursColors)
                dsa.postAnno(fid, annos)
            else:
                logging.warning(f"No annotation found in {fid} __ {svsname}")
        except:
            logging.warning(f"Exeption in {fid} __ {svsname}")

if __name__=="__main__":
    main()
