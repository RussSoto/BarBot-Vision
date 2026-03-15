"""
Embedded Vision System - Raspberry Pi

Version: V2.0 (Face Recognition)
Extends vision_system_v1.py by adding recognition and enrollment.

Description
-----------
Extends the V1 motion + face detection system by adding face recognition,
dataset loading, and runtime enrollment of unknown faces.

Features
--------
- Motion detection using frame differencing
- Face detection using OpenCV Haar cascades
- Face recognition using 128D embeddings from face_recognition / dlib
- Dataset auto-loading at startup
- Runtime enrollment of unknown faces
- Text-to-speech alerts using pyttsx3

Author
------
Russell Soto
"""

from picamera2 import Picamera2
import cv2
import time
import pyttsx3
import face_recognition
import numpy as np
import os

#Camera & Frame Settings
FRAME_WIDTH=640
FRAME_HEIGHT=480
FRAME_FORMAT="BGR888"

#Motion detection 
MIN_MOTION_AREA=500
BLUR_KERNEL=(15,15)
MOTION_HOLD_TIME=5

#Face Detection
FACE_SCALE=1.3
FACE_NEIGHBORS=5
FACE_DETECT_SKIP=5
FACE_LOST_TIMEOUT=2.5

#Face Recognition
RECOGNITION_SKIP=15
UNKNOWN_COOLDOWN=5
FACE_CROP_SIZE=(160,160)
FACE_PADDING_RATIO=0.3
DATASET_PATH="face_dataset"

#Alert / TTS
ALERT_COOLDOWN=5

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

#Runtime State
frame_count=0
last_motion=0.0
last_face=None
face_spoken_time=0
known_face_encodings=[]
known_face_names=[]
last_face_name="Unknown"
last_unknown_time=0
last_face_seen=0

def load_face_dataset(dataset_path=DATASET_PATH):
    """
    Load known faces from the dataset into memory
    Args:
        dataset_path: Root dataset folder containing one subfolder per person
            Expected Structure :
                face_dataset/
                    PersonA/
                        img1.jpg
                        img2.jpg
                    PersonB/
                        img1.jpg
    
    Returns:
        None
    """
    global known_face_encodings,known_face_names
    
    if not os.path.exists(dataset_path):
        print("No dataset folder found.")
        return
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path,person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder,img_file)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encodings = face_recognition.face_encodings(img)
            
            if len(encodings)>0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                
                print(f"Loaded {person_name}")
                
#=============================================================================================


def preprocess_frames(previous_frame_local,current_frame_local):
    """
    Convert two frames to grayscale & apply Gaussian blur(15,15 Kernel) for motion detection
    Args:
        previous_frame_local: Earlier BGR frame as the reference/background frame
        current_frame_local: Most recent BGR frame used for motion detection & face detection
    Returns:
        blurred_previous_frame: Blurred grayscale version of previous_frame_local
        blurred_current_frame: Blurred grayscale version of current_frame_local
        grayscale_for_faces: Unblurred grayscale of current_frame_local used for face detection
    """
    previous_gray = cv2.cvtColor(previous_frame_local,cv2.COLOR_BGR2GRAY)
    blurred_previous_frame=cv2.GaussianBlur(previous_gray,BLUR_KERNEL,0)
    
    grayscale_for_faces = cv2.cvtColor(current_frame_local,cv2.COLOR_BGR2GRAY)
    blurred_current_frame=cv2.GaussianBlur(grayscale_for_faces,BLUR_KERNEL,0)
    
    return blurred_previous_frame,blurred_current_frame,grayscale_for_faces

#============================================================================================= 
    
def detect_motion(previous_blurred_local,current_blurred_local): 
    """
    Detect the motion by comparing two blurred grayscale frames.
    Checks the difference, applies threshold, & dilation to reduce noise.
    Find contours to detect motion regions
    Args:
        previous_blurred_local: Blurred grayscale reference frame
        current_blurred_local: Blurred grayscale current frame
    Returns:
        True if a motion contour larger than MIN_MOTION_AREA is found
        Otherwise False
    """
    diff=cv2.absdiff(previous_blurred_local,current_blurred_local)
    thresh=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    dilated=cv2.dilate(thresh,None, iterations=2)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour)>MIN_MOTION_AREA:
            return True
    return False

#============================================================================================= 

def detect_faces(gray_frame_local):
    """
    Detect faces in a grayscale frame using loaded Haar cascade
    Args:
        gray_frame_local: Grayscale image used for face detection
    Returns:
        A list of face bounding boxes in [x,y,w,h] format
    """
    return face_cascade.detectMultiScale(gray_frame_local, scaleFactor=FACE_SCALE,minNeighbors=FACE_NEIGHBORS)

#=============================================================================================

def alert_on_face_detection(faces_local,face_spoken_time_local):
    """
    Trigger a text-to-speech alert when a face is detected with cooldown control
    Args:
        faces_local: List of detected face boxes in (x,y,w,h) format
        face_spoken_time_local: Timestamp of the last spoken face alert    
    Returns:
        face_spoken_time_local: Updated timestamp if the alert is spoken
    """
    if len(faces_local)>0:
        
            
        if time.time()-face_spoken_time_local>ALERT_COOLDOWN:
            print("Face detected! Hello from the other side")
            engine.say("Face detected!  We've been waiting for you")
            engine.runAndWait()
            face_spoken_time_local=time.time()
    return face_spoken_time_local 

#=============================================================================================

def draw_face_box(face_box_local,frame_local,name_local):
    """
    Draw a face bounding box & label on the frame
    Args:
        face_box_local: Face bounding box in (x, y, w, h) format or None 
        frame_local: BGR frame to annotate
        name_local: Label to draw above the face box
    Returns:
        None
    """
    if face_box_local is not None: 
        x,y,w,h=face_box_local
        cv2.rectangle(frame_local,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame_local,name_local,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    return

#=============================================================================================

def crop_face(frame_local, face_box_local):
    """
    Crop a detected face with padding, resize it to a normalized size, & convert it from BGR to RGB for face_recognition
    Args:
        frame_local: Full camera frame (This is the most recent frame, labeled frame2 in main)
        face_box_local: Face bounding box in (x,y,w,h) format
    Returns:
        face_img: Cropped face image in RGB format
    """
    
    x,y,w,h = face_box_local
    
    # Face bounding box is usually too tight so padding is used to expand cropped image relative to size of face
    padding = int(FACE_PADDING_RATIO*w)
    
    #Clamp to prevent cropping outside the image
    x1 = max(0, x-padding)
    y1 = max(0,y-padding)
    x2 = min(FRAME_WIDTH,x+w+padding)
    y2 = min(FRAME_HEIGHT,y+h+padding)
                
    # Face_Recognition expects normalized input size 160x160 & RGB colorspace
    face_img = frame_local[y1:y2,x1:x2] 
    face_img = cv2.resize(face_img,FACE_CROP_SIZE) 
    face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB) 
    
    return face_img

#=============================================================================================

def get_face_embedding(face_img_local):
    """
    Generate a 128-dimensional facial embedding from a cropped face image
    Args:
        face_img_local: RGB face image (160x160)
    Returns:
        face_encoding: 128D embedding vector if a face is detected.
        otherwise None
    """
    face_encoding = face_recognition.face_encodings(face_img_local)
    
    if len(face_encoding)>0:
        return face_encoding[0]    
    
    return None

#=============================================================================================

def resolve_face_name(embedding_local):
    """
    Compare a face embedding against known faces & return the best match
    Args:
        embedding_local: 128D face embedding vector
    Returns:
        known_face_name: Matched person's name if a valid match is found
        otherwise "Unknown"
    """
    
    if len(known_face_encodings) == 0:
        return "Unknown"
    
    #Recognition Logic: Euclidean distances are computed to see how far apart embeddings (faces)
    #Matches is the conversion of distances into T/F based on a threshold (True if distance < 0.6)
    matches = face_recognition.compare_faces(known_face_encodings,embedding_local)
    distances = face_recognition.face_distance(known_face_encodings,embedding_local)
                                               
    #Name Logic: Match name to lowest distance if match is True.
    #If there are multiple matches, this ensures the name is matched to lowest distance, i.e best match
    if len(distances) == 0:
        return "Unknown"
    
    best_match_index = np.argmin(distances)
    
    if matches[best_match_index]:
        return known_face_names[best_match_index]
    
    return "Unknown"

#=============================================================================================

def enroll_new_face(face_img_local, embedding_local, last_unknown_time_local):
    """
    Prompt user to name an unknown face, save it to the dataset, and add it
    to known face encodings for future
    
    Args:
        face_img_local: Cropped RGB face image
        embedding_local: 128D face
        last_unknown_time_local: Timestamp of last unknown face prompt
    
    Returns:
        user_name: The entered name if enrollment succeeds, otherwise "Unknown"
        last_unknown_time_local: Updated cooldown timestamp    
    """
    if time.time() - last_unknown_time_local <= UNKNOWN_COOLDOWN:
        return "Unknown", last_unknown_time_local
        
        
    last_unknown_time_local = time.time()
    user_name = input("Enter Name for this Face:").strip()
    
    if user_name == "":
        return "Unknown", last_unknown_time_local
    
    #Create folder for new person (if it doesnt exist)
    person_folder = f"{DATASET_PATH}/{user_name}"
    os.makedirs(person_folder,exist_ok=True)
    
    timestamp = int(time.time())
    filename = f"{person_folder}/{timestamp}.jpg"
    
    save_img = cv2.cvtColor(face_img_local, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename,save_img)
    print(f"Saved face for {user_name} to {filename}")
    
    #Add to known faces
    known_face_encodings.append(embedding_local)
    known_face_names.append(user_name)
    print(f"Added {user_name} to known faces")
    
    return user_name,last_unknown_time_local

#=============================================================================================


load_face_dataset()

while True:
    frame1=picam2.capture_array()
    frame2=picam2.capture_array()
    
    
    
    background_frame, current_frame,gray_for_faces=preprocess_frames(frame1,frame2)
    
    if detect_motion(background_frame,current_frame):
        last_motion=time.time()
    
    faces=[]
    if time.time()-last_motion < MOTION_HOLD_TIME:
        
        if frame_count % FACE_DETECT_SKIP == 0:
            faces=detect_faces(gray_for_faces)
            
            if len(faces) > 0:
                last_face_seen = time.time()
             
            #V2 design choice: track only the first detected face for stable single-face recognition
                last_face = faces[0]
                face_img = crop_face(frame2,last_face)
                
                
                
                if frame_count % RECOGNITION_SKIP == 0:
                    embedding = get_face_embedding(face_img)
                    
                    if embedding is not None:
                        last_face_name = resolve_face_name(embedding)                     
                        if last_face_name == "Unknown":
                            print("Unknown face detected...")
                            last_face_name, last_unknown_time = enroll_new_face(face_img,embedding,last_unknown_time)
            
        
        frame_count+=1
        
        if time.time()- last_face_seen > FACE_LOST_TIMEOUT:
            last_face = None
            last_face_name = "Unknown"
        
        face_spoken_time=alert_on_face_detection(faces,face_spoken_time)
        draw_face_box(last_face,frame2,last_face_name)
        
                    
    cv2.imshow("Live Feed w/ Motion + Face Detection + Recognition",frame2)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
        
picam2.close()
cv2.destroyAllWindows()
