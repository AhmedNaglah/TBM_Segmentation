import cv2
import numpy as np

def saveContouredImageMax(im, mask, mydir):
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros(np.shape(mask), dtype=np.uint8)
    contour = max(contours, key = cv2.contourArea)
    im = cv2.drawContours(im,contour,-1,(0,255,0),1)
    cv2.fillPoly(mask2, pts =[contour], color=(255))    
    cv2.imwrite(mydir, im)
    cv2.imwrite(mydir.replace('contoured','mask'), mask2)
    return im, mask2

def saveContouredImageFiltered(im, mask, mydir, thresh):
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros(np.shape(mask), dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour)>thresh:
            im = cv2.drawContours(im,contour,-1,(0,255,0),1)
            cv2.fillPoly(mask2, pts =[contour], color=(255))    
    #cv2.imwrite(mydir, im)
    #cv2.imwrite(mydir.replace('contoured','mask'), mask2)
    return im, mask2

def detectionMetric(mask1, mask2):
    ret, binary = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    p = len(contours)
    tp=0
    for contour in contours:
        maskdummy = np.zeros(np.shape(mask1), dtype=np.uint8)
        cv2.fillPoly(maskdummy, pts =[contour], color=(255))    
        union = cv2.bitwise_and(maskdummy,mask2)
        unionsum = np.sum(union)
        if unionsum>0:
            tp+=1
    ret, binary = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    fp = len(contours)-tp
    try:
        precision = tp/(tp+fp)
        sensitivity = tp/p
        return precision, sensitivity
    except:
        return 'na', 'na'

def detectionMetricParam(mask1, mask2):
    ret, binary = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    p = len(contours)
    tp=0
    for contour in contours:
        maskdummy = np.zeros(np.shape(mask1), dtype=np.uint8)
        cv2.fillPoly(maskdummy, pts =[contour], color=(255))    
        union = cv2.bitwise_and(maskdummy,mask2)
        unionsum = np.sum(union)
        if unionsum>0:
            tp+=1
    ret, binary = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    fp = len(contours)-tp
    return tp, fp, p

def saveContouredImage (im, mask, mydir):
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(im,contours,-1,(0,255,0),1)
    cv2.imwrite(mydir, im)
    cv2.imwrite(mydir.replace('contoured','mask'), mask)
    return im

def imageShow(im):
    pass
    #cv2.imshow('image show function', im)
    #cv2.waitKey()

def getIoU(mask1, mask2):
    intersection = cv2.bitwise_and(mask1,mask2)
    union = cv2.bitwise_or(mask1,mask2)
    myim = cv2.hconcat((intersection,union))
    imageShow(myim)
    return np.sum(intersection)/np.sum(union)

def getIoUParam(mask1, mask2):
    intersection = cv2.bitwise_and(mask1,mask2)
    area_aug = np.sum(mask2)
    p = np.sum(mask1)
    union = cv2.bitwise_or(mask1,mask2)
    myim = cv2.hconcat((intersection,union))
    imageShow(myim)
    inter = np.sum(intersection)
    fp=area_aug-inter
    return inter, np.sum(union), fp, p

def getSemantic(mask1, mask2):
    intersection = cv2.bitwise_and(mask1,mask2)
    union = cv2.bitwise_or(mask1,mask2)
    notunion = cv2.bitwise_not(union)
    accuracy = (np.sum(intersection) + np.sum(notunion))/mask1.size
    area_aug = np.sum(mask2)
    p = np.sum(mask1)
    union = cv2.bitwise_or(mask1,mask2)
    myim = cv2.hconcat((intersection,union))
    imageShow(myim)
    inter = np.sum(intersection)
    fp=area_aug-inter
    return inter, np.sum(union), fp, p, accuracy

def getDice(mask1, mask2):
    intersection = cv2.bitwise_and(mask1,mask2)
    union = cv2.bitwise_or(mask1,mask2)
    ints = np.sum(intersection)
    return 2*ints/(np.sum(union)+ints)

def histogram_intersection(h1, h2):
    sm = 0
    st = 0
    for i in range(len(h1)):
        sm += min(h1[i], h2[i])*100
        st += max(h1[i], h2[i])*100
    return sm/st

def histogram_intersection_range(h1, h2, myrange):
    sm = 0
    st = 0
    for i in myrange:
        sm += min(h1[i], h2[i])*100
        st += max(h1[i], h2[i])*100
    return sm/st

def RoiHist(im, mask):
    l = np.size(mask)
    mask = np.reshape(mask, (1,l))
    nzi = np.nonzero(mask)
    s = np.size(im[:,:,0])
    im_h = np.reshape(im[:,:,0], (1,s))
    im_s = np.reshape(im[:,:,1], (1,s))
    im_v = np.reshape(im[:,:,2], (1,s))

    im_h = im_h[nzi]
    im_s = im_s[nzi]
    im_v = im_v[nzi]

    hist_h, _  = np.histogram(im_h, bins=255, range=(0,255))
    hist_s, _  = np.histogram(im_s, bins=255, range=(0,255))
    hist_v, _  = np.histogram(im_v, bins=255, range=(0,255))
    return hist_h, hist_s, hist_v

def mutual_information(hgram):
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def nmi_evaluate(hgram):
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x

    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    nzx = px > 0 # Only non-zero pxy values contribute to the sum
    nzy = py > 0 # Only non-zero pxy values contribute to the sum

    Hx = np.sum(-px[nzx]*np.log(px[nzx]))
    Hy = np.sum(-py[nzy]*np.log(py[nzy]))
    nmi =  2*(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))/(Hx+Hy)
    if nmi==None:
        return -1
    else:
        return nmi



