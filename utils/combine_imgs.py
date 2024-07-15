from PIL import Image, ImageDraw, ImageFont
import os
import re
from PIL import Image, ImageOps  # Ensure ImageOps is imported
from helper import combine_images

"""
This script combines two images side by side with captions and saves the combined image.
"""

def main():
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    hand_drawn_map_path = os.path.join(current_dir, "process_hdmaps", "hdmap_path_landmarks.png")
    frames_folder = os.path.join(current_dir, "..", "files", "extracted_frames")
    output_folder = os.path.join(current_dir, "..", "files", "output")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frame numbers and sort them
    frame_files = os.listdir(frames_folder)
    frame_numbers = [int(re.search(r'(\d+)', f).group()) for f in frame_files if re.search(r'(\d+)', f)]
    frame_numbers.sort()

    # Loop through sorted frame numbers and combine them with the hand drawn map
    for number in frame_numbers:
        print("Processing frame: ", number)
        frame_file = f"frame_{number}.jpg"
        frame_path = os.path.join(frames_folder, frame_file)
        output_path = os.path.join(output_folder, f"input_{number}.jpg")
        combine_images(hand_drawn_map_path, frame_path, "Hand drawn map", "Front view", output_path, compression_factor=0.4)

if __name__ == "__main__":
    main()
    print("Done combining images")
