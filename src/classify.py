from sklearn.externals import joblib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './libs'))

import preprocessor
import subimages
import features

import cv2
import numpy as np

import matplotlib
import matplotlib.cm as cm

def get_path(p):
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), p))

font = cv2.FONT_HERSHEY_DUPLEX

def classify(img, classes='general', model='random_forest'):

    clf = joblib.load(get_path('../models/%s/%s.joblib' % (classes, model)))
    classlist = list(clf.classes_)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(classlist) - 1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Set1)
    
    colored = img.copy()
    full_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    STROKE = int(0.015 * np.min(full_image.shape))
    
    imgw, _ = full_image.shape


    rois = subimages.extract(full_image, preprocessor.default_ensemble)
    for (cropped, cnt) in rois:
        vector = features.get(full_image, cropped, cnt)
        if(vector is False):
            continue #skip it

        x,y,w,h = cv2.boundingRect(cnt)
        pred = clf.predict([vector])[0]

        color = np.array(mapper.cmap(classlist.index(pred))[:-1]) * 255
        cv2.rectangle(colored,(x,y),(x+w,y+h),color,STROKE)


        csize = len(pred)
        #texto
        print(w/(20*csize), imgw/1000)
        tsize = max(w/(20*csize), imgw/1000)
        tweight = max(int(3*tsize), 3)

        tw, th = cv2.getTextSize(pred, font, tsize, tweight)[0]

        xpos = int(x+(w-tw)/2)
        ypos = int(5 + y+h+2*STROKE+th)

        cv2.putText(colored, pred,(xpos,ypos), font, tsize, color, tweight, cv2.LINE_AA)

    return colored