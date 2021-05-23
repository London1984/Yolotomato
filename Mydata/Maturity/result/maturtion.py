# -*- coding: utf-8 -*-

import cv2
import glob
import matplotlib.pyplot as plt

path="/home/wenjie/darknet/scripts/VOCdevkit/Maturity/"
for imgfile in glob.glob("./*.jpg"):
    img=cv2.imread(imgfile)
    name= imgfile.split("/")[-1][:-4]
    width,height,chanel = img.shape
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    histh = cv2.calcHist([img_hsv],[0],None,[180],[0,179])
    #caculate the maturity score
    mature_pixel = histh[0]+histh[1]+histh[2]+histh[3]+histh[4]+histh[5]+histh[6]+histh[7]+histh[8]+histh[9]+histh[10]+histh[175]+histh[176]+histh[177]+histh[178]+histh[179]
    halfmature_pixel =histh[11]+histh[12]+histh[13]+histh[14]+histh[15]
    raremature_pixel = histh[16]+histh[17]+histh[18]+histh[19]+histh[20]
    total_pixel = width*height
    mature_score = int((mature_pixel*110+halfmature_pixel*80+raremature_pixel*60)/total_pixel)
    #plot the result
    plt.cla()
    plt.subplot(2,1,1)
    plt.imshow(img_rgb)
    plt.title('Tomato image, maturity score:%d'%mature_score)
    plt.subplot(2,1,2)
    plt.plot(histh,label="H")
    plt.title('Hue Histogram')
    plt.xlabel('Hue')
    plt.ylabel('Pixels number')
    plt.subplots_adjust(left=None,bottom=None,right = None,top= None,wspace=1,hspace=0.5)
    plt.savefig('./histogram/'+name+'_h.jpg')

   

    