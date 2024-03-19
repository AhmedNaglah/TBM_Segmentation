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


def segmentLumen(im):
    def applyThresholdWhite(patch):

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        sensitivity = 30
        lower = np.array([0,0,255-sensitivity])
        upper = np.array([255,sensitivity,255])
        mask = cv2.inRange(hsv, lower, upper)
        
        return np.array(mask, dtype='uint8')

    def smoothBinary(mask):
        kn = 5
        iterat = 6
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
        return mask

    thresholdArea = 0.02

    mask = applyThresholdWhite(im)
    maskSmooth = smoothBinary(mask)
    mask3d = cv2.cvtColor(maskSmooth, cv2.COLOR_GRAY2RGB)

    #contours, _ = cv2.findContours(maskSmooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c_max = max(contours, key = cv2.contourArea)
    area_min = cv2.contourArea(c_max) * thresholdArea

    contours_filtered = []

    amin = 200

    for contour, hie in zip(contours, hierarchy[0]):
        area = cv2.contourArea(contour)
        if area > area_min:
            if area > amin:
                contours_filtered.append((contour, hie))

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
        "name": 'lmn', 
        "description": 'lmn',  
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

    args = parser.parse_args()

    config = vars(args)

    outdir = config['outputdir']

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    dsa = DSA(config)

    dsaFolder = DSAFolder(config)
    items = dsaFolder.items

    ftr = config['name']

    for i in range(len(items)):

        fid, svsname = items[i]

        slide = Slide(config, svsname)

        _ = slide.extractForeground2()

        patches = slide.extractPatches(config['patchSize'])

        print(len(patches))

        contoursColors = {}

        contoursColors = []

        errorN = 0
        for key in patches:
            try:
                patch = patches[key]

                contours = segmentLumen(patch)

                print(f"found # {len(contours)}")
                
                x, y = key.split('_')
                contoursColors.append({'x': int(x), 'y': int(y) , 'cnt': contours, 'color': '20, 20, 200'})
            except:
                errorN+=1
        print(errorN)
        annos = generateAnno(contoursColors)
        dsa.postAnno(fid, annos)

if __name__=="__main__":
    main()
