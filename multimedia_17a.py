import cv2
import os
from matplotlib import pyplot as plt 
from numpy import e
from numpy.ma.core import log2



cam = cv2.VideoCapture("vid.mp4") 
  
try: 
      
    # creating a folder named data storing the initial 100 frames of the video
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 

try: 
      
    # creating a folder named error_frames storing the error frames which are required.
    if not os.path.exists('error_frames'): 
        os.makedirs('error_frames') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of error') 
  
# frame counter
currentframe = 0
  
while(True): 
      
    # reading frames from video
    ret,frame= cam.read() 
  
    if (ret and currentframe<100): 
        # if video is still left and the frames are less than 100 then create images 
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converting these frames into grayscale ones
        
        cv2.imwrite(name, frame) 
  
        # increasing counter 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 

#The required function is  responsible for finding the error_frames.These frames are all stored 
#to error_frames folder
def find_error_frame(i_n,i_pn):
    en=cv2.subtract(i_n,i_pn)
    return en



def find_entropy(frame):
    n=cv2.calcHist([frame],[0],None,[256],[0,256]) #creating the frame's histogram
   
    #uncomment the two following lines in order to view the actual histogram
    #plt.plot(n)
    #plt.show()
    p=n/sum(n) #propability array
    p=(p+e)#add the float constant e in order to prevent dividing with zero values
    h=sum(p*log2(1/p)) #entropy in megabits
    return(h)
 
     
        

for n in range(currentframe-1):#Creating error frames
    img1=cv2.imread('./data/frame' + str(n) + '.jpg')
  
    img2=cv2.imread('./data/frame' + str(n+1) + '.jpg')
    path='./error_frames/error_frame' +str(n+1)+'.jpg'
    print ('Creating...' + path) 
    cv2.imwrite(path,find_error_frame(img2,img1))




error_frame3=cv2.imread('./error_frames/error_frame3.jpg')#the error frame 3
frame3=cv2.imread('./data/frame3.jpg')#the initial frame 3

error_frame32=cv2.imread('./error_frames/error_frame32.jpg')#the error frame 32
frame32=cv2.imread('./data/frame32.jpg')#the initial frame 32

error_frame67=cv2.imread('./error_frames/error_frame67.jpg')#the error frame 67
frame67=cv2.imread('./data/frame67.jpg')#the initial frame 67

error_frame78=cv2.imread('./error_frames/error_frame78.jpg')#the error frame 78
frame78=cv2.imread('./data/frame78.jpg')#the initial frame 78


if(find_entropy(error_frame3)<find_entropy(frame3)):#exam if the initial hypothesis is actually valid.
    print("Error frame 3 has less entropy than frame 3")
if(find_entropy(error_frame32)<find_entropy(frame32)):#exam if the initial hypothesis is actually valid.
    print("Error frame 32 has less entropy than frame 32")
if(find_entropy(error_frame67)<find_entropy(frame67)):#exam if the initial hypothesis is actually valid.
    print("Error frame 67 has less entropy than frame 67")
if(find_entropy(error_frame78)<find_entropy(frame78)):#exam if the initial hypothesis is actually valid.
    print("Error frame 78 has less entropy than frame 78")
cv2.waitKey(0)
cv2.destroyAllWindows()



    
