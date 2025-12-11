# Sign Recognition System

## Overview

> **Note:** This project was completed during my 2nd semester of my degree, at a time when I had no prior knowledge of AI models or machine learning. The approach is purely mathematical and template-based, without any deep learning.

This project is a real-time Sign Recognition System that uses mathematical techniques and template matching to recognize hand gestures from webcam input. It leverages computer vision and linear algebra to capture hand positions, normalize them, and match against saved gesture templates.

## Features
- Real-time hand tracking and gesture recognition using webcam
- Save custom hand gesture templates
- Match gestures using normalized landmark positions
- Simple, math-based approach (no deep learning required)

## Requirements

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/dependency-opencv-blue)
![MediaPipe](https://img.shields.io/badge/dependency-mediapipe-blue)
![NumPy](https://img.shields.io/badge/dependency-numpy-blue)

## Installation
1. Clone this repository:
	```bash
	git clone https://github.com/yourusername/Sign-Recognition-System.git
	cd Sign-Recognition-System
	```
2. (Recommended) Create a Python virtual environment:
	```bash
	python -m venv venv
	# Activate the virtual environment:
	# On Windows:
	venv\Scripts\activate
	# On macOS/Linux:
	source venv/bin/activate
	```
3. Install the required Python packages:
	```bash
	pip install opencv-python mediapipe numpy
	```

## Usage Guide
1. Run the main script:
	```bash
	python track_hand.py
	```
2. The webcam window will open and start tracking your hand.
3. To record a new gesture:
	- Press `r` on your keyboard.
	- Enter a name for your gesture in the terminal.
	- The system will save the current hand position as a template.
4. The recognized gesture name will be displayed in the "Gesture" window.
5. Press `ESC` to exit.

## How It Works
- The system uses MediaPipe to detect hand landmarks (key points).
- Landmarks are normalized relative to the wrist and scaled to unit size, making recognition robust to hand position and scale.
- Gestures are saved as templates in `hand_gestures.json`.
- When a hand is detected, its normalized landmarks are compared to saved templates using Euclidean distance.
- The closest matching gesture (within a tolerance) is displayed.
## Troubleshooting
- **No hand detected:** Ensure your hand is visible to the webcam and well-lit.
- **Gesture not recognized:** Try recording the gesture again, or adjust your hand position.
- **Dependencies missing:** Install required packages with `pip install opencv-python mediapipe numpy`.

## File Structure
- `track_hand.py` — Main script for gesture tracking and recognition
- `hand_gestures.json` — Saved gesture templates (created after first save)
- `README.md` — Project documentation and user guide
