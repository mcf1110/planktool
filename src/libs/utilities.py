"""
Some commom utilities
"""
import os
import matplotlib.pyplot as plt
import cv2

CV_V3 = cv2.__version__[0] == "3"
CV_V4 = cv2.__version__[0] == "4"
DEBUG = False

def image_read(path):
    """
    Simple abstraction over imread

    Parameters
    ----------
    path : string
        Path to be loaded

    Returns
    -------
    image : opencv image
    """
    if CV_V3 or CV_V4:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def find_files(folder, filetypes=['jpg'], depth=float('inf')):
    """
    Given a directory, returns a list of all files contained in its descendant directories

    Parameters
    ----------
    folder : string
        The folder to be analyzed

    filetypes : list
        The allowed filetypes to be returned

    depth : int
        How deep should the search go. 0 means only on folder, 1 on children of folder, and so onself.
        Default is infinity, which means it will go recursively.


    Returns
    -------
    transformed : list of files
    """
    entries = os.listdir(folder)
    result = []
    for e in entries:
        completePath = os.path.join(folder, e)
        if os.path.isdir(completePath) and depth > 0:
            result += find_files(completePath, filetypes=filetypes, depth=depth-1)
        elif e[-3:].lower() in filetypes:
            result.append(completePath)
    return result

def image_show_colored(img, title=''):
    """
    Dumps an colored image to the console

    Parameters
    ----------
    img: opencvImage
        The image to be shown

    title: string
        The plot title
    """
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.title(title)
    plt.imshow(img)
    plt.show()

def image_show(img, title=''):
    """
    Dumps an black and white image to the console

    Parameters
    ----------
    img: opencvImage
        The image to be shown

    title: string
        The plot title
    """
    image_show_colored(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), title)
