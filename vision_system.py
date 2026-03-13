"""
Embedded Vision System - Raspberry Pi

Version: V1.0

Description
-----------
Lightweight embedded computer vision system designed to run on resource-constrained
hardware such as the Raspberry Pi. The system detects motion in the camera feed and
performs face detection when activity is present.

Features
--------
- Motion detection using frame differencing
- Face detection using OpenCV Haar cascades
- Event-driven processing to reduce unnecessary computation
- Text-to-speech alerts using pyttsx3

Author
------
Russell Soto
"""

from picamera2 import Picamera2
import cv2
import time
import pyttsx3


#Camera & Frame Settings
FRAME_WIDTH=640
FRAME_HEIGHT=480
FRAME_FORMAT="BGR888"

#Motion detection 
MIN_AREA=500
BLURKERNAL=(15,15)

#Face Detection
FACE_SCALE=1.3
FACE_NEIGHBORS=5

#Alert / TTS
COOLDOWN=5

#Initialize Camera 
picam2 = Picamera2()
config=picam2.create_preview_configuration(main={"size":(FRAME_WIDTH,FRAME_HEIGHT),"format":FRAME_FORMAT})
picam2.configure(config)
picam2.start()

#Initialize TTS Engine
engine=pyttsx3.init()
engine.setProperty("rate",150)
engine.setProperty("volume",1.0)

#Load Haar cascade for face detection 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Global Vars
frame_count=0
motion_detected = False
last_motion=0.0
last_face=None
face_spoken=0

def grayblur(frame1_local,frame2_local):
    """
    Convert two frames to grayscale and apply guassian blur(15,15 Kernal)
    Returns:(background_frame,current,grayscale_for_faces)
    """
    gray = cv2.cvtColor(frame1_local,cv2.COLOR_BGR2GRAY)
    background_frame=cv2.GaussianBlur(gray,BLURKERNAL,0)
    grayscale_for_faces = cv2.cvtColor(frame2_local,cv2.COLOR_BGR2GRAY)
    current=cv2.GaussianBlur(grayscale_for_faces,BLURKERNAL,0)
    
    return background_frame,current,grayscale_for_faces
    
def detect_motion(past_frame_local,meta_frame_local): 
    """
    Detect the motion by comparing 2 blurred grayscale frames.
    Checks the difference, applies threshold, & dilation to reduce noise.
    Find contours to detect motion regions
    Return: T/F
    """
    
    diff=cv2.absdiff(past_frame_local,meta_frame_local)
    thresh=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    dilated=cv2.dilate(thresh,None, iterations=2)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour)>MIN_AREA:
            return True
    return False

def detect_faces(gray):
    """
    Detect faces in grayscale image.
    Apply loaded Haar cascade classifier to the image.
    Classifier scans images at multiple scales to find faces.
    Returns: list of bounding boxes [x,y,w,h]
    """
    return face_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE,minNeighbors=FACE_NEIGHBORS)

def alert(faces_local,last_face_local,face_spoken_local):
    """
    Alerts when face is detected with a cooldown.
    Cooldown in between detection is 5 seconds.
    Updates face_spoken after TTS is trigger so that it does not play until cooldown is done.
    Returns:(last_face_local and face_spoken_local)
    """
    if len(faces_local)>0:
        last_face_local=faces_local[0]
            
        if time.time()-face_spoken_local>COOLDOWN:
            print("Face detected! Would you like a drink?")
            engine.say("Face detected!  Would you like a drink?")
            engine.runAndWait()
            face_spoken_local=time.time()
    return last_face_local,face_spoken_local 

def facebox(last_face_local,frame2_local):
    """Apply rectangle boxes around detected face"""
    if last_face_local is not None: 
        x,y,w,h=last_face_local
        cv2.rectangle(frame2_local,(x,y),(x+w,y+h),(255,0,0),2)
    return

while True:
    frame1=picam2.capture_array()
    frame2=picam2.capture_array()
    
    background, current_frame,gray4faces=grayblur(frame1,frame2)
    
    if detect_motion(background,current_frame):
        last_motion=time.time()
        motion_detected=True
    
    faces=[]
    if time.time()-last_motion<5:
        if frame_count%5 == 0:
            faces=detect_faces(gray4faces)
        frame_count+=1
        
        last_face,face_spoken=alert(faces,last_face,face_spoken)
        facebox(last_face,frame2) 
                    
    cv2.imshow("Live Feed w/ Motion + Face Detection",frame2)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
        
picam2.close()
cv2.destroyAllWindows()
