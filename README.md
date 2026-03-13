#Embedded Vision System (Raspberry Pi)
Lightweight embedded computer vision system for Raspberry Pi featuring motion detection, face detection, and real-time feedback.
Originally developed as the vision component for a robotics concept, this project evolved into a general-purpose embedded computer vision platform designed to run efficiently on resource-constrained hardware.
The system demonstrates how a practical computer vision pipeline can detect motion, identify faces, and provide interactive feedback while minimizing unnecessary computation.

____________________________________________________________________________

##Features
• Real-time camera streaming
• Motion detection using frame differencing
• Face detection using OpenCV Haar cascades
• Event-driven processing (face detection runs only when motion occurs)
• Bounding box visualization on detected faces
• Text-to-speech alerts using pyttsx3
• Designed for embedded hardware constraints

____________________________________________________________________________

##System Architecture
Camera Frame
↓
Frame Processing (grayscale + blur)
↓
Motion Detection
↓
Face Detection
↓
Bounding Box Rendering
↓
Optional User Feedback (Text-to-Speech)


Motion gating ensures that expensive operations like face detection only run when activity is detected. This approach helps maintain performance on embedded systems such as the Raspberry Pi.

____________________________________________________________________________

##Tech Stack
Python
OpenCV
NumPy
pyttsx3 (text-to-speech)
Raspberry Pi
PiCamera2

____________________________________________________________________________

##Hardware
Raspberry Pi
Camera Module (PiCamera2)

____________________________________________________________________________

##Installation
Clone the repository:
git clone https://github.com/RussSoto/BarBot-Vision.git
cd BarBot-Vision

Install dependencies:
pip install -r requirements.txt

____________________________________________________________________________

##Running the System
python vision_system.py

The program will open the camera feed and monitor for motion and faces in real time.

____________________________________________________________________________

##Repository Structure
vision_system.py
Main application script that runs the motion detection and face detection pipeline.

haarcascade_frontalface_default.xml
Pretrained OpenCV Haar cascade used for face detection.

requirements.txt
Python dependencies required to run the project.

README.md
Project documentation.

____________________________________________________________________________

##Project Roadmap
Version 1 — Motion + Face Detection
Current stable version.
• Motion detection
• Real-time video processing
• Face detection
• Text-to-speech alerts using pyttsx3
• Embedded-friendly vision pipeline

Version 2 — Face Recognition (In Development)
• Face recognition using embeddings
• Identity dataset system
• Runtime enrollment of new faces

Version 3 — Tracking & Optimization
• Face tracking to reduce CPU usage
• Detection event logging
• System performance improvements

____________________________________________________________________________

##Engineering Focus
This project focuses on building a practical embedded vision system, emphasizing:
• Real-time video processing
• Efficient computer vision pipelines
• Embedded hardware constraints
• Integration of machine learning libraries into working systems
The goal is not to train machine learning models from scratch but to build reliable systems that deploy existing computer vision tools effectively on edge devices.

____________________________________________________________________________

##Potential Applications
• Robotics vision systems
• Smart home monitoring
• Security systems
• Human-machine interaction
• Edge AI experimentation

____________________________________________________________________________

##License
MIT License

Built by Russell Soto
See you, space cowboy.
