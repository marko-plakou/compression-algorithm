import os
import cv2
from matplotlib import  pyplot as plt
import numpy
from numpy import e
from numpy.ma.core import log2


try: 
      
    # creating a folder named pframes storing the prediction frame
    if not os.path.exists('pframes'): 
        os.makedirs('pframes') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of error')
try: 
      
    # creating a folder named error_frame storing the error frame
    if not os.path.exists('error_frame'): 
        os.makedirs('error_frame') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of error')

source_frame=cv2.imread("frame6.jpg",0)#reading the source frame 
target_frame=cv2.imread("frame6.jpg",0)#reading the target frame 
height_per_block=16 #declaring block's height dimension which is 16 pixels since its 16x16 block
width_per_block=16 #declaring block's width dimension which is 16 pixels since its 16x16 block
frame_height=target_frame.shape[0]#finding the total height of the target frame
frame_width=target_frame.shape[1]#finding the total width of the target frame
prediction_frame=source_frame#initialization of prediction frame
def create_prediction_frame(x1,x2,y1,y2,vector_h,vector_w):#creates the prediction frame 
   
    prediction_frame[y1:y2,x1:x2]=source_frame[y1+vector_h:y2+vector_h,x1+vector_w:x2+vector_w]
   

def find_error_frame(img1,img2):#finds the error frame
    en=cv2.subtract(img1,img2)#substracts the target frame with the prediction frame
    return en



def find_entropy(frame):#returns entropy
    n=cv2.calcHist([frame],[0],None,[256],[0,256]) #creating the frame's histogram
   
    #uncomment the two following lines in order to view the actual histogram
    #plt.plot(n)
    #plt.show()
    p=n/sum(n) #propability array
    p=(p+e)#add the float constant e in order to prevent dividing with zero values
    h=sum(p*log2(1/p)) #entropy in megabits
    return(h)
   
     



   

def find_block_in_source_frame(h,w):#finds the given block in the source frame 
     source_frame_window = source_frame[h:h+height_per_block,w:w+width_per_block]
     target_frame_window=target_frame[h:h+height_per_block,w:w+width_per_block]
     source_frame_hist =numpy.histogram(source_frame_window,bins=256)[0]  
     target_frame_hist=numpy.histogram(target_frame_window,bins=256)[0]
     
     #initialization of the coordinates
     x1=0
     x2=0
     y1=0
     y2=0
     min=sum(abs(numpy.subtract(target_frame_hist,source_frame_hist)))#initialization of min
     
     for k in range(1,17):#k is 16
         #checking the perimeter to find the block which best matches the given block
        check_from_center_right=source_frame[h:h+height_per_block,w+k:w+width_per_block+k]
        check_from_center_left=source_frame[h:h+height_per_block,w-k:w+width_per_block-k]
        check_from_center_up=source_frame[h-k:h+height_per_block-k,w:w+width_per_block]
        check_from_center_down=source_frame[h+k:h+height_per_block+k,w:w+width_per_block]
        check_from_up_right=source_frame[h-height_per_block+k:h+k,w+k:w+width_per_block+k]
        check_from_up_left=source_frame[h-height_per_block+k:h+k,w-width_per_block+k:w+k]
        check_from_down_right=source_frame[h+k:h+height_per_block+k,w+k:w+width_per_block+k]
        check_from_down_left=source_frame[h+k:h+height_per_block+k,w-width_per_block+k:w+k]
        #min variable stores the sum of absolute differencies (SAD) betweent the given block (target_frame) and the blocks(source_block) which are being examined.
        min1=sum(abs(numpy.subtract(target_frame_hist, numpy.histogram(check_from_center_right,bins=256)[0])))
        min2=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_center_left,bins=256)[0])))
        min3=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_center_up,bins=256)[0])))
        min4=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_center_down,bins=256)[0])))
        min5=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_up_right,bins=256)[0])))
        min6=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_up_left,bins=256)[0])))
        min7=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_down_right,bins=256)[0])))
        min8=sum(abs(numpy.subtract(target_frame_hist,numpy.histogram(check_from_down_left,bins=256)[0])))
       
        #the substraction who gave the minimum result is the most matching 
        if(min>min1):
            min=min1
            y1=h
            y2=h+height_per_block
            x1=w+k
            x2=w+width_per_block+k
           
        if(min>min2):
            min=min2
            y1=h
            y2=h+height_per_block
            x1=w-k
            x2=w+width_per_block-k
            
        if(min>min3):
            min=min3
            y1=h-k 
            y2=h+height_per_block-k
            x1=w
            x2=w+width_per_block
            
        if(min>min4):
            min=min4
            y1=h+k 
            y2=h+height_per_block+k
            x1=w
            x2=w+width_per_block
           
        if(min>min5):
            min=min5
            y1=h-height_per_block+k
            y2=h+k
            x1=w+k
            x2=w+width_per_block+k
            
        if(min>min6):
            min=min6
            y1=h-height_per_block+k
            y2=h+k
            x1=w-width_per_block+k
            x2=w+k
            
        if(min>min7):
            min=min7
            y1=h+k
            y2=h+height_per_block+k
            x1=w+k
            x2=w+width_per_block+k
            
        if(min>min8):
            min=min8
            y1=h+k
            y2=h+height_per_block+k
            x1=w-width_per_block+k
            x2=w+k
        #Unblock the two following lines of code in order to view the motion vector for each block
        #print("The motion vectori is:")
        #print(((h-y1)-((h+height_per_block)-y2)),((w-x1)-((w+width_per_block)-x2)))
       #returns the coordinates of the most matching block in order to  create the prediction frame
        motion_vector_h=(h-y1)-((h+height_per_block)-y2)
        motion_vector_w=(w-x1)-((w+width_per_block)-x2)

        create_prediction_frame(x1,x2,y1,y2,motion_vector_h,motion_vector_w)
    




















def motion_prediction(source_frame,target_frame):#the requred function
    
    print('Processing...')
    #braking the target frame in 16x16 microblocks 
    for h in range(0,frame_height, height_per_block):
        for w in range(0,frame_width, width_per_block):
            window = target_frame[h+1:h+height_per_block,w+1:w+width_per_block]
            hist = numpy.histogram(window,bins=256)[0]      
            find_block_in_source_frame(h+1,w+1) #for each block it tries  the best matching to the source frame
    cv2.imwrite('./pframes/pframe7.jpg', prediction_frame) #it stores the prediction frame
    error_frame7a=cv2.imread('error_frame7a.jpg',0)#it reads the error frame 1 from the previous task
    cv2.imwrite('./error_frame/error_frame7b.jpg', find_error_frame(target_frame,prediction_frame)) #it stores the error frame
    error_frame7b=cv2.imread('./error_frame/error_frame7b.jpg',0)#it reads the error frame
    cv2.imshow('error frame',error_frame1b)#displays the error frame
    print("Entropy of the previous error frame:")
    print(find_entropy(error_frame7a))
    print("Entropy of the new error frame:")
    print(find_entropy(error_frame7b))
    if(find_entropy(error_frame7a)>find_entropy(error_frame7b)):
        print("The entropy of the new error frame is smaller")
    
      
       





           

#calling the required function 
motion_prediction(source_frame,target_frame)





    




