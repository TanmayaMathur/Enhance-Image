Video Enhancement and Face Recognition
Overview
This project implements a video enhancement application that utilizes various techniques such as super-resolution, deblurring, face detection, and face recognition. It includes a graphical user interface (GUI) built with PyQt5, enabling users to process videos and generate 3D face models from detected faces.

Features
Super-Resolution: Enhances the quality of low-resolution frames using the SRCNN model.
Deblurring: Applies a Wiener filter to reduce motion blur in video frames.
Face Detection: Automatically detects and saves faces from each processed frame.
Face Recognition: Compares detected faces against a database of known faces to find the best match.
3D Face Mesh Generation: Creates a 3D mesh of detected faces with smooth skin-like shading.
GUI Interface: A user-friendly interface to input video files and known face images.
Requirements
Make sure you have the following installed:

Python 3.x
OpenCV
PyTorch
torchvision
Pillow
face_recognition
PyQt5
mediapipe
NumPy
Matplotlib
SciPy

Usage
Input Video: Provide the path to the video file you want to enhance.

Known Face Images: Enter a comma-separated list of paths to known face images that will be used for recognition.

Run the Application: Click the "Enhance Video" button to start processing.

Optionally, enable the "3D Face Recognition" checkbox to generate 3D models of the recognized faces.

Output: Processed frames will be saved in the output_frames directory, and the best matched face image will be displayed in the GUI.

Code Overview
SRCNN Model
The SRCNN model is defined in model.py to perform super-resolution on low-resolution images. The model consists of three convolutional layers.

Wiener Filter
The Wiener filter is implemented in wiener_filter.py to deblur images. It takes an image, a kernel, and a noise variance as inputs and returns the deblurred image.

Face Detection and Recognition
In face_detection.py, the application uses the face_recognition library to detect and recognize faces in the processed frames. Detected faces are saved, and comparisons are made against known face encodings.

3D Face Mesh Generation
The 3D face mesh is generated using MediaPipe in face_detection.py, which processes the detected face and creates a 3D visualization based on facial landmarks.

Acknowledgments
OpenCV for computer vision functionality.
PyTorch for implementing the SRCNN model.
face_recognition for face detection and recognition.
MediaPipe for real-time face mesh generation.




