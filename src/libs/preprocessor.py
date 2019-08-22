"""
This file contains different preprocessing methods.

Here, we call "preprocessing" the image transformations that happen prior to
contour identification, as shown in the diagram:

`(Image) --preprocessing--> (Processed) --find_contours--> (Contours)`

Therefore, all methods below present different ways to transform an image into
a "contourizable" image.
"""
import cv2
import numpy as np
import utilities

def otsu(img, KERNEL_SIZE = 5):
    """ Transforms the image using otsu binarizarion method. It also applies floodFill
    to the top left corner, trying to get all closed shapes.

    Parameters
    ----------
    img : opencv image
        An image to be processed

    KERNEL_SIZE : int
        Kernel size for the "opening" morphological transformation.

    Returns
    -------
    transformed : opencv image
    """
    #Apply otsu threshold, so we can separate background(white) from foreground(black)
    _,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Remove noise from image
    kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    #Fill the holes left, so we get only closed shapes.
    copy = cv2.bitwise_not(opening) #Foreground needs to be white
    h, w = otsu.shape[:2] #Get image dimensions
    mask = np.zeros((h+2, w+2), np.uint8) #So we can make a mask that is two pixels wider and higher

    cv2.floodFill(copy, mask, (0,0), 255) # Floodfill from point (0, 0). Now, copy contains just the holes

    noholes = cv2.bitwise_not(opening) | cv2.bitwise_not(copy) # Combine the two images to get the hole-free image.

    return noholes

def otsu_triangles(img, KERNEL_SIZE = 5):
    """ Transforms the image using otsu binarizarion method, similar to the "otsu" function on this module.
    The only difference, however, is that this version draws a small triangle (5 percent of img's width) on each of the four corners,
    and applies floodFill to all of them, instead of just one.

    Parameters
    ----------
    img : opencv image
        An image to be processed

    KERNEL_SIZE : int
        Kernel size for the "opening" morphological transformation.

    Returns
    -------
    transformed : opencv image
    """

    _, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return _remove_holes_with_triangles(otsu, KERNEL_SIZE)

def canny(img):
    """ Transforms the image using canny edge recongnition method. Blurs the image
    a little as well, to make it more likely that edges touch.

    Parameters
    ----------
    img : opencv image
        An image to be processed

    Returns
    -------
    transformed : opencv image
    """
    thresh, _ = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(img, 0.3*thresh, 0.6*thresh)
    return cv2.GaussianBlur(edges,(15,15),0)

def sprinkles(img, KERNEL_SIZE=5):
    """ Temporary name. Uses the mean adaptive threshold, blurs the image and applis otsu. Then, it applies the triangle on corners method
    The intent here is to prevent stains on the background from being recognized as objects.

    Parameters
    ----------
    img : opencv image
        An image to be processed

    KERNEL_SIZE : int
        Kernel size for the "opening" morphological transformation.

    Returns
    -------
    transformed : opencv image
    """
    mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    blur = cv2.GaussianBlur(mean,(25,25),0)
    _, rebinary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return _remove_holes_with_triangles(rebinary, KERNEL_SIZE)

def _draw_single_triangle(img, points, color=(0,0,0)):
    """ Takes an image, transforms it via opening, and applies a triangle on each of the four corners.
    Then, it proceeds to do a floodfill on each of those corners.

    Parameters
    ----------
    img : opencv image
        Image to be drawn on

    points : matrix of coordinates
        Coordinates representing the points.

    color: tuple of RGB
        Color to paint the triangle (default is black)
    """
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img, [pts], color)

def _remove_holes_with_triangles(binarized, KERNEL_SIZE):
    """ Takes an image, transforms it via opening, and applies a triangle on each of the four corners.
    Then, it proceeds to do a floodfill on each of those corners.

    Parameters
    ----------
    binarized : opencv image
        An image to be processed. Must already be binarized.

    KERNEL_SIZE : int
        Kernel size for the "opening" morphological transformation.

    Returns
    -------
    transformed : opencv image
    """
    h, w = binarized.shape
    T_SIZE = 0.05 * w

    #Remove noise from image
    kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)
    opening = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    #Fill the holes left, so we get only closed shapes.
    copy = cv2.bitwise_not(opening) #Foreground needs to be white

    _draw_single_triangle(copy, [[0,0], [T_SIZE, 0], [0, T_SIZE]])
    _draw_single_triangle(copy, [[0,h], [T_SIZE, h], [0, h-T_SIZE]])
    _draw_single_triangle(copy, [[w,0], [w-T_SIZE, 0], [w, T_SIZE]])
    _draw_single_triangle(copy, [[w,h], [w-T_SIZE, h], [w, h-T_SIZE]])

    h, w = binarized.shape #Get image dimensions
    mask = np.zeros((h+2, w+2), np.uint8) #So we can make a mask that is two pixels wider and higher

    cv2.floodFill(copy, mask, (0, 0), 255)     # Top Left
    cv2.floodFill(copy, mask, (w-1, 0), 255)   # Bottom Left
    cv2.floodFill(copy, mask, (0, h-1), 255)   # Top Right
    cv2.floodFill(copy, mask, (w-1, h-1), 255) # Bottom Right

    return cv2.bitwise_not(opening) | cv2.bitwise_not(copy) # Combine the two images to get the hole-free image.


def project(img):
    """Preprocessor made for Computer Vision classes"""
    h, w = img.shape[:2]
    
    #Detect dark bg
    s = np.mean(np.hstack([img[0, :], img[h-1, :], img[:, 0], img[:, w-1]])) #mean of border
    if np.mean(img) - s > 10:
        img = cv2.bitwise_not(img)
    
    size = int(min(img.shape) / 2)
    if size % 2 == 0:
        size +=1
    kernel_size = max(int(size/60), 5) #at least 5 pixels
    
    img = cv2.medianBlur(img, kernel_size * 2 + 1)
    
    #Thresholding
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,size,1)
    
    #Opening and closing
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    close = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    
    #floodFill
    th, im_th = cv2.threshold(opening, 220, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = im_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    
    T_SIZE = kernel_size * 10
    _draw_single_triangle(im_floodfill, [[0,0], [T_SIZE, 0], [0, T_SIZE]])
    _draw_single_triangle(im_floodfill, [[0,h], [T_SIZE, h], [0, h-T_SIZE]])
    _draw_single_triangle(im_floodfill, [[w,0], [w-T_SIZE, 0], [w, T_SIZE]])
    _draw_single_triangle(im_floodfill, [[w,h], [w-T_SIZE, h], [w, h-T_SIZE]])
    
    im_floodfill[0, :] = 0
    im_floodfill[h-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, w-1] = 0
    
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)     # Top Left
    
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    return im_out

def new_process(img):
    """A new trial using median blur"""
    k = np.array(img.shape).max()
    percent_of_k = lambda x: int((k//(200/x))*2+1)
    
    MEDIAN_SIZE = percent_of_k(2)
    THRESH_SIZE = percent_of_k(4)
    CLOSE_SIZE = percent_of_k(0.5)
    OPEN_SIZE = percent_of_k(1)
    
    median = cv2.medianBlur(img, MEDIAN_SIZE)
    th = cv2.adaptiveThreshold(median, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,max(3, THRESH_SIZE),2)
    #blur = cv2.GaussianBlur(th,(THRESH_SIZE,THRESH_SIZE),0)
    op = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((CLOSE_SIZE, CLOSE_SIZE)))
    op = cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((OPEN_SIZE, OPEN_SIZE)))
    
    return _remove_holes_with_triangles(th, 5)

def new_process_2(img):
    """Another trial using median blur"""
    k = np.array(img.shape).max()
    percent_of_k = lambda x: int((k//(200/x))*2+1)
    
    MEDIAN_SIZE = percent_of_k(2)
#    MEDIAN_2_SIZE = percent_of_k(0.25)
    CLOSE_SIZE = percent_of_k(0.5)
    OPEN_SIZE = percent_of_k(1.7)
#    DIST_SIZE = percent_of_k(0.7)
#    
    equ = cv2.equalizeHist(img)
    median = cv2.medianBlur(equ, MEDIAN_SIZE)
#    laplacian = cv2.Laplacian(median.astype(np.uint8),cv2.CV_8UC1)
#    median = cv2.medianBlur(laplacian, MEDIAN_2_SIZE)
#    _,th3 = cv2.threshold(median,1,255,cv2.THRESH_BINARY)
#    op = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, np.ones((OPEN_SIZE, OPEN_SIZE)))
#    op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, np.ones((CLOSE_SIZE, CLOSE_SIZE)))
        
    _,th3 = cv2.threshold(median,.18*255,255,cv2.THRESH_BINARY)
    op = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, np.ones((CLOSE_SIZE, CLOSE_SIZE)))
    op = cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((OPEN_SIZE, OPEN_SIZE)))
#    utilities.image_show(_remove_holes_with_triangles(op, 5))
    
    return _remove_holes_with_triangles(op, 5)


def stacked(img):
    """Combining new_process and new_process_2"""
    k = np.array(img.shape).max()
    percent_of_k = lambda x: int((k//(200/x))*2+1)
    OPEN_SIZE = percent_of_k(2)
    
    a, b = new_process(img), new_process_2(img)
    ret = a|b
    op = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, np.ones((OPEN_SIZE, OPEN_SIZE)))
    return (op)

def ensemble(methods):
    """Given a list of preprocessors, generates an ensemble with them"""
    size = len(methods)
    th = (255*size) * .2
    def pp(img):
        ret = np.sum([m(img) for m in methods], axis=0) <= th
        ret = 255*(ret.astype(np.uint8))
        
        return _remove_holes_with_triangles(ret, 5)
    return pp
        
"""The default ensemble, using 6 preprocessors"""
default_ensemble = ensemble([new_process_2, new_process, project, sprinkles, canny, otsu_triangles])