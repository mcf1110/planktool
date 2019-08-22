# -*- coding: utf-8 -*-
"""
Responsible for finding the regions of interest (subimages) on a given image.
"""
import cv2
import utilities as utils

def find_contours(image):
    """ Given an image, it finds all the contours on it.
    Just an abstraction over cv2.findContours

    Parameters
    ----------
    image : opencv image
        An image to be processed.

    Returns
    -------
    contours : opencv contours
    """
    if utils.CV_V3:
        _, contours, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contour_list(image, preprocessed, MIN_FILTER=3000):
    """ Given an image and its preprocessed version, returns the cropped image and its contours.

    The return value is in the format: [(CroppedImage, Contour)]

    Parameters
    ----------
    image : opencv image
        The original unprocessed image

    preprocessed: opencv image
        The processed image

    MIN_FILTER : int
        Contours with an area lower than this value are discarded
    
    MAX_FILTER_PERCENT: float
        Contours with dimensions that exceed this percentage of the image will be discarded

    Returns
    -------
    result : array of tuples
    """
    contours = find_contours(preprocessed) #gets contours in the preprocessed image
    result = []
    
    if utils.CV_V3 or utils.CV_V4:
        orb = cv2.ORB_create()
    else:
        orb = cv2.ORB()
    kp = orb.detect(image, None)
    
    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        
        has_keypoint = any([cv2.pointPolygonTest(cnt, k.pt, False) > -1 for k in kp])
        if not has_keypoint:
            continue
                
        if(c_area > MIN_FILTER): #FILTERING MIN SIZE
            if utils.DEBUG : print(cv2.contourArea(cnt))
            (x,y),r = cv2.minEnclosingCircle(cnt)
            (x,y, r) = (int(max(r,x)), int(max(r,y)), int(r))
            
            #FILTERING MAX SIZE
            #if r > MAX_FILTER_PERCENT*image.shape[1] or r > MAX_FILTER_PERCENT*image.shape[0]:
                #continue
            
            (y1,y2,x1,x2) = (y-r,y+r,x-r,x+r)
            result.append( (image[y1:y2,x1:x2], cnt) )
    return result

def extract(img, preproc):
    """
    The method to be used outside this module. Takes an image and a preprocessing
    method, and return a list of tuples whose first position is the cropped image,
    and second position is the contour that generated it. Visually, it takes this
    format: [(CroppedImage, Contour)]

    Parameters
    ----------
    img : opencv image
        The image to be processed.

    preproc : function
        A function that will process the image, i.e., one of the functions available
        in the "preprocessor" module.

    Returns
    -------
    result : array of tuples
    """
    if utils.DEBUG: utils.image_show(preproc(img))
    return get_contour_list(img, preproc(img))
