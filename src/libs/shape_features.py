"""
Responsible for generating the shape features of a given image, which are:

- Rectangle Mean (0-255): The average intensity of the bounding rectangle

- Ellipse Mean (0-255): The average intensity of the bounding ellipse.

- Aspect Ratio (0-1): MajorAxis/MinorAxis, how similiar are the image dimensions. The closer to 0, the more "streched" the image is.

- Area: Area of the object.

- Area Hull: Area of the object's convex hull.

- Solidity: Area/Area hull.

- Extent: Area/Bounding box area.

- Perimeter: Perimiter of the object.

- Perimeter Hull: Perimiter of the object's convex hull.

- Circularity: Perimiter of a circle "C" with the same area as the object.

- Heywood Circularity: Perimeter of the object / Circumference of "C"

- Waddel Circularity: Diameter of "C"

- Rectangularity: Area of object/Area of bouding rectangle

- Eccentricity: sqrt(major_axis^2 - minor_axis^2)/major_axis

- Ellipse Area: Area of the bounding ellipse
"""

import cv2
import mahotas
import numpy as np
import sys
import matplotlib.pyplot as plt
from os.path import (basename)
import utilities

def dump(img, noholes, otsu, cnt, hull, name, area):
    SHOW = True
    truename = basename(name)[:-4]
    truename = truename + " (area = %s)" % area
    print('./hullDump/%s' % (truename))
    def display(im, suffix):
        if SHOW:
            plt.title(name)
            plt.imshow(im); plt.show()
        else:
            cv2.imwrite('./hull_dump/%s %s.jpg' % (truename, suffix), im)

    hulled = img.copy()
    cv2.drawContours(img, [cnt], 0, (0,255, 0), 4)
    cv2.drawContours(hulled, [hull], 0, (0,255,0), 4)

    display(otsu, 'otsu')
    display(noholes, 'no holes')
    display(img, 'contours')
    display(hulled, 'covex hull')

    sys.exit()

def get_labels ():
    """
    Generates the labels.

    Returns
    -------
    list : the list of labels
    """
    return ["rectangle_mean", "ellipse_mean", "aspect_ratio", "area", 
            "area_hull", "solidity", "extent", "perimiter", "perimeter_hull", 
            "circularity","heywood_circularity", "waddel_circularity", 
            "rectangularity", "eccentricity", "ellipseArea",
            "convexity2", "convexity3"] + ["hu%d"%d for d in range(7)] + \
            ["har%d"%d for d in range(13)]

def crop_box(image, rect):
    """
    Given an opencv Image and a RotatedRect, returns a cropped version of the image.

    Parameters
    ----------
    image : opencv Image
        An image to be cropped

    rect : opencv RotatedRect
        A box which defines the image's crop area

    Returns
    -------
    cropped : opencv image
    """
    if utilities.CV_V3 or utilities.CV_V4:
        box = cv2.boxPoints(rect)
    else:
        box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]; H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs); x2 = max(Xs)
    y1 = min(Ys); y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle+=90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int((x2-x1)),int((y2-y1)))

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H; croppedH = H if not rotated else W
    return cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))

def get_rect_features(image, contour):
    """
    Calculates features regarding the minimum area rectangle enclosing the contour.

    Parameters
    ----------
    image : opencv image
        An image to be processed.

    contour : opencv contour
        The contour of the object.

    Returns
    -------
    rect_mean : float
        Mean intensity of pixels in the rectangle
    ratio : float
        Aspect ratio of the rectangle (major/minor)
    minor_axis : float
        Length of the minor axis
    major_axis : float
        Length of the minor axis
    """
    rect = cv2.minAreaRect(contour)
    croppedRotated = crop_box(image, rect)
    height, width = croppedRotated.shape[:2]
    large = np.max([height, width])
    small = np.min([height, width])
    return (np.mean(croppedRotated), np.float(small)/large, small ,large)

def get_el_mean(image, contour):
    """
    Calculates features regarding the ellipse enclosing the contour.

    Parameters
    ----------
    image : opencv image
        An image to be processed.

    contour : opencv contour
        The contour of the object.

    Returns
    -------
    ellipse_mean : float
        Mean intensity of pixels inside the ellipse, decreased of the pixels outside of it
    area : float
        Area of the ellipse
    """
    ellipse = cv2.fitEllipse(contour)
    croppedRotated = crop_box(image, ellipse)
    height, width = croppedRotated.shape[:2]
    ellipse_area = np.pi * height/2 * width/2
    centerx, centery = (width/2,height/2)
    # We invert the pixels outside of the ellipse, so we can penalize them
    for x in range(width):
        for y in range(height):
            if np.float(x-centerx)**2/(centerx)**2 + np.float(y-centery)**2/(centery)**2 > 1:
                croppedRotated[y,x] = 255 - croppedRotated[y,x]
    return np.mean(croppedRotated), ellipse_area

def get(image, contour):
    """
    Calculates all of the shape features of the image

    Parameters
    ----------
    image : opencv image
        An image to be processed.

    contour : opencv contour
        The contour of the object.

    Returns
    -------
    features : an array of features
    """
    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)
    equivalent_area_circle_r = np.sqrt(area/np.pi)
    equivalent_perimiter_circle_r = 0.5 * perimeter/np.pi

    # area / area of a circle with the same perimeter
    compactness = area / (np.pi * equivalent_perimiter_circle_r**2)
    # perimeter / perimeter of a circle with the same area
    heywood_circularity = perimeter / (2 * np.pi * equivalent_area_circle_r)
    # diameter of a circle with the same area
    waddel_circularity = 2 * equivalent_area_circle_r
    

    rect_mean, aspect_ratio, _minor_axis, _major_axis = get_rect_features(image, contour)
    rectangularity = area / (_minor_axis*_major_axis)
    eccentricity = np.sqrt(_major_axis**2 - _minor_axis**2)/_major_axis

    hull = cv2.convexHull(contour)
    hull_area, hull_perimeter = cv2.contourArea(hull), cv2.arcLength(hull, True)

    # CONVEXITIES
    solidity = float(area)/hull_area # aka convexity_1
    convexity_2 = hull_perimeter / float(perimeter)
    convexity_3 = 2*(_minor_axis+_major_axis) / float(perimeter)
    
    # Calculate Moments and Hu Moments
    moments = cv2.moments(image)
    huMoments = np.array([(-1) * np.copysign(1.0, h) * np.log10(abs(h)) for h in cv2.HuMoments(moments)]).flatten()
    
    _, _, _bw, _bh = cv2.boundingRect(contour)
    bounding_area = _bw * _bh
    extent = float(area)/bounding_area
    
    try:
        el_mean, ellipse_area = get_el_mean(image, contour)
    except: # Sometimes it won't get an ellipse
        el_mean, ellipse_area = 0, 0
        
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)
    
    mean, std = cv2.meanStdDev(image, mask=mask)
    mean, std = (mean[0][0], std[0][0])
    
    haralick = mahotas.features.haralick(image).mean(0)
    #zernike = mahotas.features.zernike_moments(image, 1)
        
    return [rect_mean, el_mean, aspect_ratio, area, hull_area, solidity, 
            extent, perimeter, hull_perimeter, compactness,
            heywood_circularity, waddel_circularity, 
            rectangularity, eccentricity, ellipse_area,
            
            convexity_2, convexity_3] + list(huMoments) + list(haralick)
