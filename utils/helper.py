from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageOps  # Ensure ImageOps is imported
import re
import os

class IterationLogger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self._initialize_log_file()

    def _initialize_log_file(self):
        with open(self.log_file_path, "w") as log_file:
            log_file.write("Starting Iteration Logs\n")
            log_file.write("------------------------------------------------\n")

    def log_iteration(self, iteration_index, question, response_info, action_question, action_output):
        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"Iteration Index: {iteration_index}\n")
            log_file.write(f"Question: {question}\n")
            log_file.write("LOCALIZATION RESPONSE:\n")
            log_file.write(f"Response Info: {response_info}\n")
            log_file.write("------\n")
            log_file.write(f"Navigation Question: {action_question}\n")
            log_file.write(f"Action Output: {action_output}\n")
            log_file.write("------------------------------------------------\n")

def add_caption(image, text, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size=40):
   
    # Increase the space for the caption if needed
    caption_space = 100  # Adjusted from 60 to 100 for more space
    new_image = Image.new("RGB", (image.width, image.height + caption_space), "white")
    new_image.paste(image, (0, 0))

    # Create a drawing context
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Default font used, as specified font was not found.")

    # Calculate text width and height using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # right - left
    text_height = text_bbox[3] - text_bbox[1]  # bottom - top

    # Calculate position for the text to be centered
    text_x = (new_image.width - text_width) / 2
    text_y = image.height + (60 - text_height) / 2

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return new_image


def downsize_image(image, compression_factor, verbose = True):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    new_width = int(original_width * compression_factor)
    new_height = int(original_height * compression_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if verbose:
        print(f"Original size: {original_width}x{original_height} pixels")
        print(f"Compressed size: {new_width}x{new_height} pixels")

    return resized_image


def combine_images(image_path1, image_path2, caption1, caption2, output_path, padding=10, border=5, compression_factor=0.5, verbose=False):
    # Open the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Convert img1 to grayscale and then to a binary image (white is the background)
    img1_bw = img1.convert("L").point(lambda x: 0 if x<128 else 255, '1')
    
    # Find the bounding box of non-white (non-background) pixels in img1
    bbox = img1_bw.getbbox()
    
    # Crop img1 to the bounding box if it is not None (i.e., the image is not entirely white)
    if bbox:
        img1 = img1.crop(bbox)

    # Determine the target height (the taller of the two images)
    target_height = max(img1.height, img2.height)

    # Resize img1 to have the target height, maintaining aspect ratio
    aspect_ratio_img1 = img1.width / img1.height
    new_img1_width = int(target_height * aspect_ratio_img1)
    # img1 = img1.resize((new_img1_width, target_height), Image.ANTIALIAS)
    img1 = img1.resize((new_img1_width, target_height), Image.Resampling.LANCZOS)
    # Resize img2 to have the target height, maintaining aspect ratio
    aspect_ratio_img2 = img2.width / img2.height
    new_img2_width = int(target_height * aspect_ratio_img2)
    img2 = img2.resize((new_img2_width, target_height), Image.Resampling.LANCZOS)

    # Add captions to images
    img1_with_caption = add_caption(img1, caption1)
    img2_with_caption = add_caption(img2, caption2)

    # Add black borders to images and captions
    img1_with_caption = ImageOps.expand(img1_with_caption, border=border, fill='black')
    img2_with_caption = ImageOps.expand(img2_with_caption, border=border, fill='black')

    # New image width and height with padding and border adjustments
    dst_width = img1_with_caption.width + img2_with_caption.width + 3 * padding + 2 * border
    dst_height = img1_with_caption.height + 2 * padding  # Adjusted to include caption height and borders

    # Create a new image with white background and extra space for padding
    combined_img = Image.new('RGB', (dst_width, dst_height), "white")

    # Paste img1 and img2 side by side with padding
    combined_img.paste(img1_with_caption, (padding + border, padding + border))
    combined_img.paste(img2_with_caption, (img1_with_caption.width + 2 * padding, padding + border))

    # Crop the combined image to remove excess white space
    # Find the bounding box of non-white pixels
    bbox = combined_img.getbbox()
    if bbox:
        combined_img = combined_img.crop(bbox)

    # Save the combined image
    resized_combined_img = downsize_image(combined_img, compression_factor, verbose)
    resized_combined_img.save(output_path)


def get_sorted_indices_from_directory(directory_path):
    """
    Scans the given directory for files matching the pattern 'input_{idx}.jpg',
    extracts the indices, and returns them sorted in ascending order.
    
    Parameters:
    - directory_path (str): The path to the directory to scan.
    
    Returns:
    - List[int]: A list of indices sorted in ascending order.
    """
    # Compile the regex pattern to match files of the form 'input_{idx}.jpg'
    pattern = re.compile(r'input_(\d+)\.jpg')
    
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Extract indices from files matching the pattern
    indices = [int(pattern.match(f).group(1)) for f in files if pattern.match(f)]
    
    # Sort the indices in ascending order
    sorted_indices = sorted(indices)
    
    return sorted_indices