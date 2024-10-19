import cv2
import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.optim as optim
import numpy as np
import face_recognition
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# SRCNN Model Definition
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Function to load image and preprocess
def preprocess_image(image_path, upscale_factor):
    img = Image.open(image_path).convert('YCbCr')
    y, cb, cr = img.split()
    img_resized = y.resize((int(y.width * upscale_factor), int(y.height * upscale_factor)), Image.BICUBIC)
    img_input = ToTensor()(img_resized).unsqueeze(0)
    return img_input, cb, cr

# Post-processing to convert back to an image
def postprocess_image(output_tensor, cb, cr):
    output_img_y = output_tensor.squeeze(0).clamp(0, 1)
    output_img_y = ToPILImage()(output_img_y)
    cb_resized = cb.resize(output_img_y.size, Image.BICUBIC)
    cr_resized = cr.resize(output_img_y.size, Image.BICUBIC)
    final_img = Image.merge('YCbCr', [output_img_y, cb_resized, cr_resized]).convert('RGB')
    return final_img

# Super-Resolution function
def super_resolve_image(model, image_path, upscale_factor=2):
    img_input, cb, cr = preprocess_image(image_path, upscale_factor)
    with torch.no_grad():
        output = model(img_input)
    return postprocess_image(output, cb, cr)

# Wiener Filter for Deblurring
def wiener_filter(image, kernel, noise_var):
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    image_ft = np.fft.fft2(image)

    # Calculate the Wiener filter
    wiener_filter = np.conj(kernel_ft) / (np.abs(kernel_ft) ** 2 + noise_var)

    # Apply the filter
    deblurred_ft = wiener_filter * image_ft
    deblurred_image = np.fft.ifft2(deblurred_ft)

    return np.abs(deblurred_image)

# Function to deblur an image
def deblur_image(image_path, kernel, noise_variance):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    deblurred_image = wiener_filter(image, kernel, noise_variance)
    return deblurred_image

# Function to detect and save faces in an image
def detect_and_save_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_images = []  # List to store paths of saved face images

    for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        face_image_path = f"face_{i + 1}.jpg"
        cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        face_images.append(face_image_path)  # Store the path of the saved face image

    return face_images, face_locations  # Return paths of saved face images and locations

# Load known faces and their names
known_face_encodings = []  # List to store face encodings of known faces
known_face_names = []      # List to store names of known faces

# Load known face images and encode them
def load_known_faces(image_paths):
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(image_path.split('/')[-1])  # Use filename as name

# Function to compare a face with known faces
def find_best_match(face_image_path):
    unknown_image = face_recognition.load_image_file(face_image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    
    best_match_index = -1
    highest_accuracy = 0
    
    for i, known_encoding in enumerate(known_face_encodings):
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        accuracy = 1 - distance  # Convert distance to similarity
        
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_match_index = i

    return known_face_names[best_match_index], highest_accuracy

# Video to frames with super resolution, deblurring, face detection, and face recognition
def video_to_best_matched_face(video_path, output_folder, model, kernel, noise_variance, known_faces, ui, upscale_factor=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    load_known_faces(known_faces)  # Load known faces before processing

    highest_accuracy = 0
    best_face_image_path = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_frame_path = os.path.join(output_folder, f'temp_frame_{frame_count:04d}.jpg')
        cv2.imwrite(temp_frame_path, frame)

        super_resolved_image = super_resolve_image(model, temp_frame_path, upscale_factor)
        super_resolved_path = os.path.join(output_folder, f'super_resolved_frame_{frame_count:04d}.jpg')
        super_resolved_image.save(super_resolved_path)

        deblurred_image = deblur_image(super_resolved_path, kernel, noise_variance)
        output_image_path = os.path.join(output_folder, f'frame_{frame_count:04d}_final.jpg')
        cv2.imwrite(output_image_path, deblurred_image.astype(np.uint8))

        # Display processed frame in the UI
        ui.display_processed_frame(deblurred_image.astype(np.uint8))

        # Detect faces in the deblurred image
        detected_faces_images, face_locations = detect_and_save_faces(output_image_path)

        # Compare each detected face with known faces
        for face_image_path in detected_faces_images:
            best_match, accuracy = find_best_match(face_image_path)
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_face_image_path = face_image_path  # Update the best matched face image path

        # Cleanup temporary files
        os.remove(temp_frame_path)
        os.remove(super_resolved_path)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames to '{output_folder}'.")

    if best_face_image_path:
        return best_face_image_path, highest_accuracy  # Return the best matched face image path and accuracy
    else:
        return None, None  # No matches found

# Function to generate 3D face mesh
def generate_3d_face_mesh(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        
        h, w, _ = image.shape
        face_3d = []
        for landmark in landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z  # z is relative depth
            face_3d.append([x, y, z])
        
        face_3d = np.array(face_3d)
        
        points_2d = face_3d[:, :2]  # Take only x, y for triangulation
        tri = Delaunay(points_2d)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        z_values = face_3d[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_normalized = (z_values - z_min) / (z_max - z_min)  # Normalize z for color mapping
        
        base_skin_color = np.array([1.0, 0.78, 0.66])  # Light skin tone RGB
        
        for simplex in tri.simplices:
            triangle = face_3d[simplex]
            
            mean_z = np.mean(triangle[:, 2])
            color_intensity = (mean_z - z_min) / (z_max - z_min)
            
            shading_factor = (1 - color_intensity) * 0.5
            skin_shaded = base_skin_color * (1 - shading_factor)
            
            poly = Poly3DCollection([triangle], color=skin_shaded, edgecolor='none')
            ax.add_collection3d(poly)
        
        ax.scatter(face_3d[:, 0], face_3d[:, 1], face_3d[:, 2], c=base_skin_color, s=2)
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
        
        plt.title('3D Face Model with Smooth Skin-Like Shading')
        plt.show()

    face_mesh.close()

# Example usage
class VideoEnhancementApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancement with Face Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.video_input_label = QtWidgets.QLabel("Video Input:")
        self.layout.addWidget(self.video_input_label)
        
        self.video_input_line_edit = QtWidgets.QLineEdit()
        self.layout.addWidget(self.video_input_line_edit)
        
        self.known_faces_label = QtWidgets.QLabel("Known Face Images (comma-separated):")
        self.layout.addWidget(self.known_faces_label)
        
        self.known_faces_line_edit = QtWidgets.QLineEdit()
        self.layout.addWidget(self.known_faces_line_edit)
        
        self.enhance_button = QtWidgets.QPushButton("Enhance Video")
        self.layout.addWidget(self.enhance_button)
        
        self.three_d_checkbox = QtWidgets.QCheckBox("3D Face Recognition")
        self.layout.addWidget(self.three_d_checkbox)
        
        self.processed_video_label = QtWidgets.QLabel("Processed Video Output:")
        self.layout.addWidget(self.processed_video_label)
        
        self.enhance_button.clicked.connect(self.process_video)

        self.setLayout(self.layout)

    def display_processed_frame(self, frame):
        h, w, _ = frame.shape
        qt_image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.processed_video_label.setPixmap(pixmap)

    def process_video(self):
        video_path = self.video_input_line_edit.text()
        known_faces = [face.strip() for face in self.known_faces_line_edit.text().split(',')]
        
        output_folder = 'output_frames'
        model = SRCNN()  # Load pre-trained or initialize model

        # Define the kernel (Gaussian kernel in this case)
        kernel_size = 5
        sigma = 1.0
        x = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(x, x)

        # Noise variance (adjust based on your image)
        noise_variance = 0.1

        best_face_image_path, best_accuracy = video_to_best_matched_face(video_path, output_folder, model, kernel, noise_variance, known_faces, self)

        if best_face_image_path:
            print(f"Best matched face image: {best_face_image_path}, Accuracy: {best_accuracy:.2f}")
            if self.three_d_checkbox.isChecked():
                generate_3d_face_mesh(best_face_image_path)
        else:
            print("No matched faces found.")

# Run the application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = VideoEnhancementApp()
    window.show()
    sys.exit(app.exec_())
