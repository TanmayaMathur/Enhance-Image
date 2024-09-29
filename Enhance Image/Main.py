import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

# Initialize the Super Resolution Model (ESPCN)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = "D:/Imp/OneDrive/Desktop/Ethos/Enhance Image/td.pb"

# Check if the model file exists before loading
if os.path.isfile(model_path):
    print("Model file exists. Attempting to load model...")
    sr.readModel(model_path)  # Load the pre-trained super resolution model
    sr.setModel("espcn", 4)   # Set the model to use ESPCN with 4x upscaling
    print("Model loaded successfully.")
else:
    print(f"Error: Model file '{model_path}' does not exist. Please check the file path.")
    exit(1)  # Exit if the model file is not found

def enhance_image(image):
    """Enhance the input image using super-resolution and other techniques."""
    result = sr.upsample(image)  # Apply super-resolution to upscale the image
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Kernel for sharpening
    result = cv2.filter2D(result, -1, kernel_sharpen)  # Apply sharpening filter

    # Reduce noise in the image
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)

    # Improve contrast using histogram equalization
    img_yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Deblur the image using a Gaussian filter
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result

# Load the input image for enhancement
image_path = "D:/Imp/OneDrive/Desktop/Ethos/Enhance Image/Photo.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
else:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format for face detection
    results = faceDetection.process(imgRGB)  # Process the image to detect faces

    if results.detections:  # If faces are detected
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box  # Get bounding box of detected face
            ih, iw, ic = img.shape  # Get image dimensions
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            face = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]  # Extract face region

            # Enhance the extracted face image
            enhanced_face = enhance_image(face)

            # Resize the enhanced face to match the original bounding box size
            enhanced_face_resized = cv2.resize(enhanced_face, (bbox[2], bbox[3]))
            img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]] = enhanced_face_resized  # Replace face in the original image

            # Draw bounding box and confidence score on the original image
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(detection.score[0] * 100)}%",
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                2,
            )

    # Save the enhanced image for evaluation
    cv2.imwrite("enhanced_image.jpg", img)

    # Display the original and enhanced images side by side for comparison
    img_comparison = np.hstack((cv2.imread(image_path), img))
    cv2.imshow("Comparison", img_comparison)

    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows


