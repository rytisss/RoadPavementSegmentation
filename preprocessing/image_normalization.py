import cv2
import numpy as np

def adaptive_histogram_normalization(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # -----Reading the image-----------------------------------------------------
    #cv2.imshow("img", img_bgr)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    #cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(320, 320))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    #cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    final_grey = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('final', final_grey)
    #cv2.waitKey(1)
    return final_grey