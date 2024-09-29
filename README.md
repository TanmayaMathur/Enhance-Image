Video Deblurring and Enhancement using Wiener Filter
Overview
This project focuses on enhancing the clarity of blurred videos using a combination of advanced image processing techniques. Each frame of a video is processed to reduce blur, remove noise, and improve visual quality through the following steps:

Deblurring using Wiener deconvolution.
Noise Reduction via KMeans clustering.
Detail Enhancement using OpenCV's detailEnhance function.
The project also provides a quantitative comparison of the deblurring process through a graph, which shows the percentage of clear pixels in the original, deblurred, and enhanced frames.

Features
Wiener Deconvolution: Used for image and video deblurring.
KMeans Clustering: Reduces noise after deblurring.
Image Enhancement: Adds more detail to the processed video.
Clear Pixel Analysis: Calculates and displays the percentage of clear pixels in the original, deblurred, and enhanced videos.
Frame-by-Frame Video Processing: Efficient handling of video input for real-time or batch processing.

Dependencies
Ensure you have the following Python packages installed:

OpenCV (cv2) - for video frame manipulation and image processing.
NumPy - for numerical operations.
scikit-learn - for KMeans clustering.
matplotlib - for visualizing the graph of pixel clarity.

Usage
Update the paths in the video_deblurring.py script:

Set input_video_path to the location of your input video file.
Set output_video_path to your desired output file location.
Run the script:

bash
Copy code
python video_deblurring.py
The script will process each frame of the video by performing the following operations:

Deblurring: Apply Wiener deconvolution to each frame.
Noise Reduction: Apply KMeans clustering to reduce noise.
Detail Enhancement: Use OpenCV's detailEnhance function to improve frame clarity.
Pixel Analysis: A graph will be generated to compare the percentage of clear pixels in the original, deblurred, and enhanced frames.
The output video will be saved to the specified output_video_path.

Example
Input:
A blurry, noisy video file located at input_video.mp4.
Output:
A processed video saved to output_video.mp4, which has improved clarity, reduced noise, and enhanced details.
Graph:
A bar graph showing the percentage of clear pixels for the original, deblurred, and enhanced versions of the video.
Code Explanation
Deblurring: The Wiener filter uses a predefined Point Spread Function (PSF) to reverse the blurring effect by applying frequency domain transformations (FFT).

Clustering: KMeans is used to reduce the noise by clustering similar pixel values together, which helps in simplifying the image data and removing small artifacts.

Enhancement: cv2.detailEnhance() enhances the processed image by refining edges and making important details more visible.

Pixel Analysis: The script calculates the percentage of "clear" pixels (pixels with intensity above a threshold) in each processed frame and visualizes the results using matplotlib.

Customization
Adjusting the Wiener Filter: You can tweak the psf (Point Spread Function) and noise variance (noise_var) in the deblur_image function to improve results for different types of blurring.
KMeans Clustering: The number of clusters (n_clusters) can be adjusted for more or less aggressive noise reduction.

Potential Improvements
Implement parallel processing for faster video processing.
Introduce options for different deblurring techniques.
Add support for real-time video stream deblurring.

![c](https://github.com/user-attachments/assets/ad405873-886e-4c33-a5a9-94b25282f122)


