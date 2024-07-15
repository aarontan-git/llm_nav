import cv2
import os
import re

"""
This script extracts frames from a video at a given interval and saves them as images.
"""

def extract_frames(video_path, output_folder, interval_seconds=1):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video as a float
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    
    # Calculate the intervals at which frames will be captured
    interval_frames = int(fps * interval_seconds)
    frame_indices = [i * interval_frames for i in range(total_frames // interval_frames + 1)]
    
    for i in frame_indices:
        # Set the current frame position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame at the current position
        ret, frame = cap.read()
        
        # If frame reading was successful, save the frame
        if ret:
            cv2.imwrite(os.path.join(output_folder, f"frame_{i}.jpg"), frame)
        else:
            print(f"Error: Could not read frame {i}.")
    
    # Release the video capture object
    cap.release()
    print("Frame extraction complete.")


def extract_all_frames(video_path, output_folder):
    """
    Extracts every frame from a video and saves them as images in the specified output folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    
    for i in range(total_frames):
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame at the current position
        ret, frame = cap.read()
        
        # If frame reading was successful, save the frame
        if ret:
            cv2.imwrite(os.path.join(output_folder, f"frame_{i}.jpg"), frame)
        else:
            print(f"Error: Could not read frame {i}.")
    
    # Release the video capture object
    cap.release()
    print("All frame extraction complete.")


def images_to_video(image_folder, output_video_path, fps=3):
    """
    Combines all images in a folder into an MP4 video file with the specified FPS.
    """
    # Get all image files from the folder, sorted to maintain order
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

    # Custom sort function to extract and convert numerical parts of the filename to integers
    def sort_key(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else filename

    # Get all image files from the folder, sorted numerically to maintain order
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")], key=sort_key)
    # print("images: ", images)

    # Read the first image to determine the video size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    total_images = len(images)
    for idx, image in enumerate(images):
        frame = cv2.imread(os.path.join(image_folder, image))
        out.write(frame)  # Write the frame to the video
        print(f"Processing image {idx + 1}/{total_images}")

    # Release everything when job is finished
    out.release()
    print("Video creation complete.")


# Example usage
# video_path = "/home/asblab/aaron/files/gazebo_test.mov"
# output_folder = "/home/asblab/aaron/files/extracted_frames_scenes/scene_9924"
# output_folder = "/home/asblab/aaron/files/gazebo_frames"
output_folder = "/home/asblab/aaron/files/hardware_trials/trial_1_images"

# extract_frames(video_path, output_folder)
# extract_all_frames(video_path, output_folder)
images_to_video(output_folder, "trial_1.mp4", fps=15)

