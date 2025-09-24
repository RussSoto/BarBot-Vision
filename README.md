# BarBot-Vision 

BarBot-Vision is a Raspberry Pi powered computer vision system that detects motion, identifies faces, and greets guest with text-to-speech(TTS). It acts as the eys and voice of your BarBot.


---

## Requirements 
-Raspberry Pi ( tested on Pi 3B+ with Picamera Module 2 )
-Python 3.X
-Libriaes 
	-pip install opencv-python pyttsx3
	-(Picamera2 comes w/ Raspberry Pi OS Bookworm or later, or install via 'sudo apt install python2-picamera2')

---
	
## Features
-Motion Detection using OpenCV
-Face detection with Haar cascades 
-Text-to-Speech guest greetings
-Live Camera feed with face boxing
-Lightweight & optimized for Raspberry Pi

---

## Run It 
Clone the repository and run the script:
	git clone https://github.com/TBD//BarBot-Vision.git
	cd BarBot-Vision
	python3 MotionAndFace.py
	
Press 'q' t quit the live feed

---

## Project Structure 

BarBot-Vision/
|
|--MotionAndFace.py # Main Program
|--haarcascade_frontalface_default.xml
|--README.md
 --requirements.txt


## Roadmap
- Face recognition for known guests 
-Bottle Detection and classification for drink selection 
-Voice-controlled drink suggestions
-Integration with BarBot (robotic bartender arm)
- Web dashboard for live monitoring and control 
-Event logging for guest interactions and drink history 

---

## License

MIT License- free to use and modify .

---

## Acknowledgments

Powered by Raspberry Pi and OpenCv. 
Built with curiosity and creativity. 
- Space Cowboy 
