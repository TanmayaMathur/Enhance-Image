import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load image
image_path = 'test3.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get face landmarks
results = face_mesh.process(image_rgb)

# Check if landmarks are detected
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0]
    
    # Convert landmarks to 3D coordinates
    h, w, _ = image.shape
    face_3d = []
    for landmark in landmarks.landmark:
        x = landmark.x * w
        y = landmark.y * h
        z = landmark.z  # z is relative depth
        face_3d.append([x, y, z])
    
    face_3d = np.array(face_3d)
    
    # Perform Delaunay triangulation to generate faces
    points_2d = face_3d[:, :2]  # Take only x, y for triangulation
    tri = Delaunay(points_2d)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the z-values for color shading
    z_values = face_3d[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min)  # Normalize z for color mapping
    
    # Set a base skin tone
    base_skin_color = np.array([1.0, 0.78, 0.66])  # Light skin tone RGB
    
    # Plot the filled triangles with varying colors based on depth (z)
    for simplex in tri.simplices:
        triangle = face_3d[simplex]
        
        # Calculate the average depth of the triangle
        mean_z = np.mean(triangle[:, 2])
        color_intensity = (mean_z - z_min) / (z_max - z_min)
        
        # Adjust the color based on depth and blend it with the base skin tone
        shading_factor = (1 - color_intensity) * 0.5  # Adjust shading intensity
        skin_shaded = base_skin_color * (1 - shading_factor)  # Apply shading to skin tone
        
        poly = Poly3DCollection([triangle], color=skin_shaded, edgecolor='none')  # Remove lines
        ax.add_collection3d(poly)
    
    # Optionally, plot the 3D face landmarks as points using the same color
    ax.scatter(face_3d[:, 0], face_3d[:, 1], face_3d[:, 2], c=base_skin_color, s=2)
    
    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
    
    plt.title('3D Face Model with Smooth Skin-Like Shading')
    plt.show()

# Release resources
face_mesh.close()

