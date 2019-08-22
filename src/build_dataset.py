"""
Computes the features of each region of interest (subimage) in each of the images of INPUT
and produces a csv file.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './libs'))

import subimages
import features
import os
import pandas as pd
import utilities as utils
import preprocessor

import cv2

INPUT = '../input_images'
OUTPUT = './'

def build_dataset():
    directories = [x[0] for x in os.walk(INPUT)]
    directories.reverse()

    #Clear existing
    existing_csv = utils.find_files(OUTPUT, filetypes=['csv'])
    for csv in existing_csv:
        os.remove(csv)

    matrix = []

    for d in directories:
        print ("Getting features for %s" % d)

        files = utils.find_files(d, depth=0) #all jpegs within d

        for f in files:
            full_image = utils.image_read(f)
            rois = subimages.extract(full_image, preprocessor.default_ensemble)
            i = 0
            for (cropped, cnt) in rois:
                vector = features.get(full_image, cropped, cnt)
                if(vector is False):
                    continue #skip it
                specific = os.path.basename(os.path.dirname(f))
                general = os.path.dirname(f).split(os.sep)[1]
                filename = os.path.normpath(f) #"%s_%d" % ( os.path.basename(f), i )
                vector += [specific, general, filename] + list(cv2.boundingRect(cnt))
                matrix.append(vector)
                
                i += 1
    cols = features.get_labels() + ['specific_class', 'general_class', 'filename', 'x', 'y', 'w', 'h']
    df = pd.DataFrame(matrix, columns=cols)

    df.to_csv(os.path.join(OUTPUT, 'dataset.csv'))