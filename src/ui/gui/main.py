import tkinter as tkr
import tkinter.filedialog as dialog
import os.path as path
from PIL import Image, ImageTk

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../libs'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from classify import classify
import cv2

def save_file():
    global last_location, saveImg
    path = dialog.asksaveasfilename(defaultextension='.jpg',
                                initialdir=last_location,
                                filetypes = [('jpg', '.jpg')])
    if path:
        cv2.imwrite(path, saveImg)

def select_image(path):
    global panelA, panelB, saveBtn, saveImg
    image = cv2.imread(path)
    
    classified = classify(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    saveImg = classified
    
    classified = cv2.cvtColor(classified, cv2.COLOR_BGR2RGB)

    height = min(500, image.shape[0])
    width = int(image.shape[1]/image.shape[0] * height)
    
    # convert the images to PIL format...
    image = Image.fromarray(image)
    classified = Image.fromarray(classified)
    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image.resize((width, height),Image.ANTIALIAS))
    classified = ImageTk.PhotoImage(classified.resize((width, height),Image.ANTIALIAS))

    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = tkr.Label(image=image, width=width, height=height)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)

        # while the second panel will store the edge map
        panelB = tkr.Label(image=classified, width=width, height=height)
        panelB.image = classified
        panelB.pack(side="right", padx=10, pady=10)
        
        
        saveBtn = tkr.Button(tk, text='Save image', command=save_file)
        saveBtn.pack(side="bottom", padx="10", pady="10")

    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image, width=width, height=height)
        panelB.configure(image=classified, width=width, height=height)
        panelA.image = image
        panelB.image = classified
        
    global btn
    btn.configure(text='Load image')

def askFile():
    global last_location, btn
    file_path = dialog.askopenfilename(
                    initialdir = last_location,
                    title = "Select image",
                    filetypes = [("jpeg","*.jpg")])
    if file_path:
        btn.configure(text='Loading image...')
        last_location = path.dirname(file_path)
        tk.after(100, lambda: select_image(file_path))

last_location = path.expanduser("~")
panelA = None
panelB = None
saveBtn = None
saveImg = None

tk = tkr.Tk()
tk.title('Planktool')
btn = tkr.Button(tk, text='Load image', command=askFile)
btn.pack(side="bottom", padx="10", pady="10")

tk.minsize(300, 150)
tk.mainloop()
