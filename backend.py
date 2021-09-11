from defter import backend
import matplotlib.pyplot as plt
from colorizers import *
from tkinter import *
import tkinter.filedialog
import shutil
import os

backend.init(path='src', extensions=['frontend.js'])

# load colorizers
img_path = None
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()


@backend.expose
def get_path():
    global img_path
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    img_path = tkinter.filedialog.askopenfilename(title='上传图片',
                                                  filetypes=[('All Files', '*')])
    root.destroy()
    shutil.copyfile(img_path, "src/in.png")
    return img_path


@backend.expose
def trans(path):
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    plt.imsave(os.getcwd() + '/src/out.png', out_img_siggraph17)
    print(os.getcwd() + '/src/out.png')
    return os.getcwd() + '/src/out.png'


# trans(r"C:\Users\starb\Desktop\colorization-master\imgs\img.png")

# TODO Coding here :-)


backend.start('frontend.html', mode="chrome", size=(935, 665))
