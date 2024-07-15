import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import supervision as sv
import cv2
import math

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import io

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """
    Loads a model from the Hugging Face Hub based on the specified repository ID and checkpoint files.

    Parameters:
    - repo_id (str): The repository ID on Hugging Face Hub where the model and configuration files are stored.
    - filename (str): The name of the file containing the model's state dictionary.
    - ckpt_config_filename (str): The name of the configuration file for the model.
    - device (str, optional): The device to load the model onto ('cpu' by default).

    Returns:
    - model: The loaded model, ready for inference, with weights loaded from the checkpoint file.
    """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   


def open_det_inference(image_dir, filename, groundingdino_model, TEXT_PROMPT, save_directory=None):
    """
    Performs object detection on an image using a specified model and text prompt, annotates the image with the detected objects, and optionally saves the annotated image.

    Parameters:
    - image_dir (str): Directory path where the image file is located.
    - filename (str): Name of the image file.
    - groundingdino_model (model): Pre-loaded model used for detection.
    - TEXT_PROMPT (str): Text prompt to guide the detection.
    - save_directory (str, optional): Directory path where the annotated image should be saved. If None, the image is not saved.

    Returns:
    - tuple: A tuple containing two lists:
        1. A list of detection results: [boxes_tensor, logits, phrases]
            - boxes_tensor (torch.Tensor): Coordinates of the bounding boxes.
            - logits (torch.Tensor): Confidence scores of the detections.
            - phrases (list of str): Detected object labels.
        2. A list of image data: [image_source, annotated_frame, image]
            - image_source (numpy.ndarray): Original image data as a NumPy array.
            - annotated_frame (numpy.ndarray): Annotated image data as a NumPy array.
            - image (PIL.Image.Image): Image loaded for model prediction.
    """
    # Construct the full path to the image using image_dir and filename
    image_path = os.path.join(image_dir, filename)
    image_ = Image.open(image_path)
    image_source, image = load_image(image_path)

    boxes_tensor, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=0.36, 
        text_threshold=0.25
    )

    annotated_frame, boxes_np = annotate(image_source=image_source, boxes=boxes_tensor, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Save the annotated image if a save directory is provided
    # if save_directory is not None:
    #     annotated_image_filename = f"annotated_{filename}"
    #     Image.fromarray(annotated_frame).save(os.path.join(save_directory, annotated_image_filename))

    return [boxes_tensor, logits, phrases], [image_source, annotated_frame, image_]


def open_seg_inference(sam_predictor, image_source, filename, annotated_image, boxes, save_directory):
    """
    Performs segmentation on an image using a specified predictor and bounding boxes, overlays the segmentation masks on the annotated image, and saves the result.

    Parameters:
    - sam_predictor (object): The SAM predictor used for segmentation.
    - image_source (numpy.ndarray): The original image data as a NumPy array.
    - filename (str): The filename of the image being processed.
    - annotated_image (numpy.ndarray): The annotated version of the image.
    - boxes (torch.Tensor): Tensor containing bounding box data.
    - save_directory (str): Directory path where the annotated image with masks should be saved.

    Returns:
    - tuple: A tuple containing:
        1. annotated_frame_with_all_masks (numpy.ndarray): The image with all the masks overlaid.
        2. masks (torch.Tensor): The bool segmentation masks generated for the bounding boxes.
    """

    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Set image in the SAM predictor (adds about 4 GB to the GPU)
    sam_predictor.set_image(image_source)

    # Convert normalized box format from cx, cy, w, h to unnormalized x1, y1, x2, y2
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    # Transform boxes to the target device and image dimensions
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda:0')

    # Predict segmentation masks for the transformed boxes
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Overlay all masks on the annotated frame
    annotated_frame_with_all_masks = overlay_all_masks(masks, annotated_image)

    # Extract index from the input file name to match the output file name
    input_file_index = filename.split('_')[-1].split('.')[0]

    # Save the annotated image with masks
    output_image_path = os.path.join(save_directory, f'mask_{input_file_index}.jpg')
    Image.fromarray(annotated_frame_with_all_masks).convert("RGB").save(output_image_path)
    print(f"SAVING MASKED IMAGE AT {output_image_path} ...")

    return annotated_frame_with_all_masks, masks


def classify_boxes_by_quadrant(detection_result, image_width, image_height, middle_width_percent=33):
    """
    Classifies each box in boxes_np into left, front, or right quadrant of the image and updates the dictionary with the quadrant information.
    
    Parameters:
    - detection_result: List of dictionaries, where each dictionary contains the keys 'phrase', 'box', and 'logit'.
      The 'box' key should have bounding box coordinates in the format (cx, cy, w, h).
    - image_width: Width of the image.
    - image_height: Height of the image.
    - middle_width_percent: Percentage width of the middle/front section. Defaults to 33%.

    Returns:
    - None: The function updates the input list of dictionaries in place.
    """
    # Calculate the width of the middle section
    middle_width = image_width * (middle_width_percent / 100)
    side_width = (image_width - middle_width) / 2

    # Iterate over each box in boxes_np
    for idx, box_info in enumerate(detection_result):
        box = box_info['box']

        # Calculate the center x-coordinate of the box
        center_x = box[0] * image_width

        # Classify the box based on its center x-coordinate
        if center_x < side_width:
            quadrant = 'left'
        elif center_x < side_width + middle_width:
            quadrant = 'front'
        else:
            quadrant = 'right'
        
        # Add the quadrant information to the dictionary
        box_info['quadrant'] = quadrant
    return detection_result


def filter_and_describe_objects(detection_result_quadrants, threshold=0.37):
    """
    Filters out objects detected with logits less than a certain threshold and
    generates descriptions of the remaining objects and their locations.

    Parameters:
    - detection_result_quadrants (list): List of dictionaries, where each dictionary contains:
        - 'phrase' (str): Detected object label.
        - 'box' (tuple): Bounding box coordinates.
        - 'logit' (float): Confidence score of the detection.
        - 'quadrant' (str): Quadrant classification of the object ('left', 'front', 'right').
    - threshold (float): Logit threshold for filtering objects. Defaults to 0.37.

    Returns:
    - str: A concatenated description of the objects and their locations.
        - Example: "Door to the right, Stairs to the front, Window to the left"
    """
    descriptions = []
    left_turn_detected = False
    right_turn_detected = False

    for item in detection_result_quadrants:
        logit = item['logit']
        phrase = item['phrase']
        quadrant = item['quadrant']
        if logit >= threshold and phrase.lower() != "floor":
            if phrase.lower() == "left_turn":
                left_turn_detected = True
            elif phrase.lower() == "right_turn":
                right_turn_detected = True
            else:
                descriptions.append(f"{phrase} to the {quadrant}")


    if left_turn_detected and right_turn_detected:
        descriptions.append("Both left and right turn are detected")
    elif left_turn_detected:
        descriptions.append("Left turn detected")
    elif right_turn_detected:
        descriptions.append("Right turn detected")
    
    return ', '.join(descriptions)


def make_single_prediction(image_path, text_prompt, save_directory='/home/asblab/aaron/files/detection/'):
    """
    Makes a single prediction on the given image with the specified text prompt and saves the output image.

    Parameters:
    - image_path (str): Path to the image file.
    - text_prompt (str): Text prompt for the prediction.
    - save_directory (str, optional): Directory to save the output image. Defaults to '/home/asblab/aaron/files/detection/'.
    """
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=0.35, 
        text_threshold=0.25
    )

    annotated_frame_tuple = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame_tuple[0]  # Assuming the first element is the image array
    annotated_frame = annotated_frame[...,::-1]  # BGR to RGB

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    input_file_index = image_path.split('_')[-1].split('.')[0]  # Extract index from the input file name
    output_filename = f'annotated_{input_file_index}.jpg'
    output_path = os.path.join(save_directory, output_filename)
    Image.fromarray(annotated_frame).save(output_path)
    print(f"Saved annotated frame to {output_path}")

    return image_source, annotated_frame, [boxes, logits, phrases]


def overlay_all_masks(masks, annotated_frame):
    # Start with the annotated frame
    final_frame = Image.fromarray(annotated_frame).convert("RGBA")
    
    # Iterate through all masks
    for i in range(masks.shape[0]):
        mask = masks[i][0]  # Get the current mask
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)  # Random color for each mask
        h, w = mask.shape[-2:]
        mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        
        # Overlay the current mask on the final frame
        final_frame = Image.alpha_composite(final_frame, mask_image_pil)
    
    return np.array(final_frame)


def segment_and_save_with_masks(image_source, filename, annotated_image, boxes, predictor, save_directory=None):
    """
    This function processes an image by segmenting it according to specified bounding boxes, 
    applying segmentation masks to an annotated version of the image, 
    and then saving the modified image. 
    The saved file's name incorporates an index derived from the original file name 
    to facilitate easy identification.

    Parameters:
    - image_source (numpy.ndarray): The original image data as a NumPy array.
    - filename (str): The filename of the image being processed.
    - annotated_image (numpy.ndarray): The annotated version of the image.
    - boxes (torch.Tensor): Tensor containing bounding box data.
    - predictor (object): Model predictor used for segmentation.
    - save_directory (str, optional): Directory where the resulting image with overlaid masks will be saved.

    Returns:
    - masks (torch.Tensor): The segmentation masks generated for the bounding boxes.
    """
    # Ensure the save directory exists
    if save_directory is not None:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    # Set image in the SAM predictor (adds about 4 GB to the GPU)
    predictor.set_image(image_source)

    # Convert normalized box format from cx, cy, w, h to unnormalized x1, y1, x2, y2
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    # Transform boxes to the target device and image dimensions
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda:0')

    # Predict segmentation masks for the transformed boxes
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # print("Masks: ", masks)

    # Overlay all masks on the annotated frame
    annotated_frame_with_all_masks = overlay_all_masks(masks, annotated_image)

    if save_directory is not None:
        print("Saving image")

        # Extract index from the input file name to match the output file name
        input_file_index = filename.split('_')[-1].split('.')[0]
        # Save the annotated image with masks
        output_image_path = os.path.join(save_directory, f'mask_{input_file_index}.jpg')
        Image.fromarray(annotated_frame_with_all_masks).convert("RGB").save(output_image_path)
        # print(f"Annotated image with masks saved at {output_image_path}")

    # print("MASKS SHAPE: ", masks.shape)
    # print("MASKS: ", masks)
    # print()

    return masks


def extract_mask(mask_output, item, filename, save_directory=None):
    """
    Identifies and extracts a specific mask or combines multiple masks from a collection based on a provided text prompt. 
    The combined or single extracted mask is then saved as a JPEG image in the specified directory. 

    Parameters:
    - mask_output (list of tuples): List of tuples where each tuple contains a mask tensor and its corresponding label.
    - item (str): The specific phrase used to identify the corresponding mask.
    - filename (str): The base filename for the output image, used when saving the mask.
    - save_directory (str, optional): The directory path where the output image will be saved. If None, the image is not saved.

    Returns:
    - mask (torch.Tensor): The extracted or combined mask corresponding to the given text prompt.
    - mask_image (PIL.Image.Image, optional): The saved image of the extracted or combined mask, if save_directory is provided.
    """

    if save_directory is not None:
        # Ensure the save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    # Find all indices of the label that contains 'floor'
    indices = [i for i, (mask_array, label) in enumerate(mask_output) if item in label]
    if not indices:
        print(f"No labels containing 'floor' found in mask_output.")
        return

    # Ensure all masks are on the same device (e.g., CPU) before combining
    device = 'cpu'  # or 'cuda:0' if you prefer to work on GPU
    combined_mask = torch.zeros_like(torch.tensor(mask_output[0][0])).to(device)
    for index in indices:
        mask_to_combine = torch.tensor(mask_output[index][0]).to(device)
        combined_mask = torch.logical_or(combined_mask, mask_to_combine)

    if save_directory is not None:
        print("Saving mask")
        # Convert the combined mask to a PIL image and save it
        mask_image = Image.fromarray((combined_mask.cpu().numpy() * 255).astype(np.uint8))
        output_path = os.path.join(save_directory, f'{filename}_mask_{item.replace(" ", "_")}.jpg')
        mask_image.save(output_path)
        print(f"Saved combined mask for '{item}' to {output_path}")
        return combined_mask, mask_image
    else:
        mask_image = None

    return combined_mask, mask_image


def highlight_edges(binary_tensor, save_path=None):
    """
    Identifies all the pixels along the edges in a binary tensor and highlights these pixels with red.

    Parameters:
    - binary_tensor (torch.Tensor): A binary image tensor with values True/False.
    - save_path (str): Path to save the output image with highlighted edges.

    Returns:
    - edges (numpy.ndarray): A binary edge map where edge pixels are 255 and non-edge pixels are 0.
    """
    # Convert binary tensor to numpy array and then to uint8 type
    image = (binary_tensor.numpy() * 255).astype(np.uint8)

    # Create a color image to draw red on edges
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Edge detection using Canny
    # Perform edge detection using the Canny algorithm
    # The thresholds for the hysteresis procedure are set to 50 and 150
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Highlight edges with red
    color_image[edges != 0] = [0, 255, 0]  # Set edge pixels to yellow

    if save_path is not None:
        # Save the result
        result_image = Image.fromarray(color_image)
        result_image.save(save_path)
        print(f"Image with highlighted edges saved at {save_path}")

    return edges


def extract_line_segments(edges, minlinelength, maxlinegap):
    """
    This function detects and extracts line segments from a binary edge map using the Hough Transform.

    Parameters:
    - edges (numpy.ndarray): A binary edge map where edge pixels are 255 and non-edge pixels are 0.
    - minlinelength (int): Minimum number of pixels making up a line. Lines shorter than this are rejected.
    - maxlinegap (int): Maximum allowed gap between points on the same line to link them.

    Returns:
    - lines (numpy.ndarray): An array of detected lines, where each line is represented by the coordinates of its endpoints.
    """
    # Ensure the edges are in uint8 format
    edges_uint8 = np.uint8(edges)

    # Line detection using Hough Transform
    # edges_uint8: The input binary edge map in uint8 format.
    # 1: The distance resolution of the accumulator in pixels.
    # np.pi / 180: The angle resolution of the accumulator in radians.
    # threshold=50: The minimum number of intersections to detect a line.
    # minLineLength=minlinelength: The minimum number of pixels making up a line.
    # maxLineGap=maxlinegap: The maximum gap between two points to be considered in the same line.
    lines = cv2.HoughLinesP(edges_uint8, 1, np.pi / 180, threshold=15, minLineLength=minlinelength, maxLineGap=maxlinegap)
    # for gazebo, threshold was set to 25
    # for real world, threshold was set to 15

    return lines


def get_edges(image_source, mask_output, filename, save_directory=None):
    """
    Processes an image to detect and overlay line segments on the floor area. It first segments the image to identify the 'floor' area,
    then highlights the edges within this area, extracts line segments from these edges, and finally overlays these lines on the original annotated image.

    Parameters:
    - image_source (numpy.ndarray): The original image data as a NumPy array.
    - filename (str): The filename of the image being processed.
    - mask_output (list of tuples): List of tuples where each tuple contains a mask array and its corresponding label.
    - save_directory (str, optional): Directory where the resulting image with overlaid lines will be saved.

    Returns:
    - lines (list): List of detected line segments, each represented as a tuple of endpoints.
    """
    item = 'floor'
    # Extract the mask corresponding to the 'floor' area
    # mask is a tensor of shape torch.Size([h, w])
    mask, mask_image = extract_mask(mask_output, item, filename)

    # Highlight the edges within the 'floor' mask
    # edges is a np array of shape (h, w), where edges are 255 and non-edges are 0
    edges = highlight_edges(mask, save_path="/home/asblab/aaron/s3/segmented/lines/edges.jpg")

    # Extract line segments from the highlighted edges
    lines = extract_line_segments(edges, minlinelength=10, maxlinegap=100)

    # Create a copy of the image to draw lines on
    image_with_lines = image_source.copy()
    # Draw each line on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color with thickness 2
    # Save the image with lines
    Image.fromarray(image_with_lines).save("/home/asblab/aaron/s3/segmented/lines/image_with_lines.jpg")
    print(f"Image with lines saved")

    return lines


def categorize_lines(lines, threshold=10, image_height=None, min_line_length=50):
    """
    Categorizes lines into vertical, horizontal, positive slope, and negative slope,
    excluding horizontal lines near the bottom border of the image,
    and lines shorter than a specified minimum length.

    Parameters:
    - lines (np.ndarray): Array of lines, each represented as [[x1, y1, x2, y2]].
    - threshold (int): Threshold for determining if a line is vertical or horizontal.
    - image_height (int, optional): Height of the image to exclude horizontal lines near the bottom.
    - min_line_length (int): Minimum length of a line to be considered for categorization.

    Returns:
    - dict: Dictionary with keys 'vertical', 'horizontal', 'positive_slope', 'negative_slope' containing lists of line coordinates.
    """
    categories = {
        "vertical": [],
        "horizontal": [],
        "positive_slope": [],
        "negative_slope": []
    }

    bottom_margin = 5  # Margin from the bottom of the image to exclude horizontal lines
    # For gazebo,m bottom_margin = 80 to get rid of itself
    # For real world, bottom_margin = 5

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length < min_line_length:
            continue  # Skip lines shorter than the minimum length
        elif abs(y1 - y2) <= threshold:
            if image_height is not None and (y1 > image_height - bottom_margin and y2 > image_height - bottom_margin):
                continue  # Skip horizontal lines near the bottom border
            categories["horizontal"].append((x1, y1, x2, y2))
        elif abs(x1 - x2) <= threshold:
            categories["vertical"].append((x1, y1, x2, y2))
        else:
            if (x2 - x1) * (y2 - y1) > 0:
                categories["negative_slope"].append((x1, y1, x2, y2))
            else:
                categories["positive_slope"].append((x1, y1, x2, y2))

    return categories


def draw_lines_on_image(image_dir, filename, categories, save_directory, line_types_to_draw):
    """
    Draws specified types of categorized lines on the original image and saves it as a single image with different colors for each category. Additionally, for 'right_turn' and 'left_turn', a circle blob is overlaid at the midpoint of the horizontal line.

    Parameters:
    - image_dir (str): Directory path where the original image is located.
    - filename (str): The filename of the image to be processed.
    - categories (dict): Dictionary containing categorized lines with their respective coordinates.
    - save_directory (str): The directory where the modified image should be saved.
    - line_types_to_draw (list): List of line types to draw, e.g., ['right_turn', 'left_turn']. Specifies which categories of lines to draw on the image.

    Returns:
    - None: The function saves the modified image directly to the specified directory and does not return any value.
    """

    image_path = os.path.join(image_dir, filename)

    # Load the original image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    if image is None:
        print("Error: Image not found.")
        return

    # Colors for each category
    colors = {
        "vertical": (255, 255, 0),  # Yellow, Gradio: Teal 
        "horizontal": (0, 165, 255),  # BLUE, Gradio: Yellow
        "positive_slope": (255, 0, 0),  # RED (LEFT SIDE), Gradio: Blue
        "negative_slope": (0, 0, 255),  # GREEN (RIGHT SIDE), Gradio: Green
    }

    mask_and_label_list = []
    turn_bounding_boxes = {}

    # Draw only the specified types of lines
    for category in line_types_to_draw:
        # print("category: ", category)
        lines = categories.get(category, []) # get the lines for the category
        # print("lines: ", lines)
        if category in ["right_turn", "left_turn"]:
            # print("category: ", category)
            if len(lines) >= 1:
                # Calculate midpoints for all lines
                mid_points = [( (line[0] + line[2]) // 2, (line[1] + line[3]) // 2) for line in lines]
                # print("mid_points: ", mid_points)

                # Calculate the average midpoint
                avg_x = sum(point[0] for point in mid_points) // len(mid_points)
                avg_y = sum(point[1] for point in mid_points) // len(mid_points)

                # draw text on the image
                text = "LEFT" if category == "left_turn" else "RIGHT"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                if category == "left_turn":
                    text_x = avg_x  # Start text at midpoint for LEFT
                else:
                    text_x = avg_x - text_size[0]  # Adjust starting x-coordinate for RIGHT
                # cv2.putText(image, text, (text_x, avg_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

                # Generate a binary 2D mask array for the circle to be displayed in Gradio GUI
                mask_array = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.circle(mask_array, (avg_x, avg_y + 15), radius=10, color=1, thickness=-1)
                # Append the mask array and turn label to a list
                turn_label = "LEFT" if category == "left_turn" else "RIGHT"
                mask_and_label = (mask_array, turn_label)
                # Assuming you have a list to store these tuples
                mask_and_label_list.append(mask_and_label)

                # Get bounding box coord for the circle (format: cx, cy, w, h)
                cx, cy = avg_x, avg_y + 15
                w, h = 20, 20  # Since radius is 10, width and height are 2 * radius
                turn_bounding_box = torch.tensor([[cx/image_width, cy/image_height, w/image_width, h/image_height]])
                turn_bounding_boxes[turn_label] = turn_bounding_box # Create a dictionary to store bounding boxes for each turn label

            else:
                print("No lines found for category: ", category)
                turn_bounding_box = None
        else:
            # Draw other types of lines normally
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(image, (x1, y1), (x2, y2), colors[category], 2)

    # Construct the full path to save the modified image
    filename = filename.replace('.jpg', '_lines.jpg')
    output_path = os.path.join(save_directory, filename)
    cv2.imwrite(output_path, image)
    print(f"SAVING IMAGE WITH LINES AT {output_path} ...")

    return image, mask_and_label_list, turn_bounding_boxes


def find_turns(image_dir, filename, lines_dict, radius):
    """
    Identifies and annotates potential turning points on an image by detecting intersections between horizontal and vertical lines within a specified radius. 
    The function updates the provided dictionary with new keys ('left_turn', 'right_turn') 
    based on the relative positions of the intersecting lines and their proximity to the image borders.

    Parameters:
    - image_dir (str): Directory path where the original image is located.
    - filename (str): The filename of the image being analyzed.
    - lines_dict (dict): A dictionary containing categorized lines with keys 'vertical' and 'horizontal', each associated with a list of line endpoints.
    - radius (float): The maximum distance between endpoints of horizontal and vertical lines to consider them as intersecting.

    Returns:
    - dict: The updated dictionary with additional keys for each detected pair of nearby lines, indicating potential turning points.
    """
    image_path = os.path.join(image_dir, filename)

    def is_within_radius(x1, y1, x2, y2, radius):
        """Check if the distance between two points is within the specified radius."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) <= radius

    vertical_lines = lines_dict.get('vertical', [])
    horizontal_lines = lines_dict.get('horizontal', [])

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return lines_dict

    image_width = image.shape[1]
    image_center_x = image_width / 2
    border_threshold = 30  # Threshold to consider a line touching the border

    if horizontal_lines and vertical_lines:
        for h_line in horizontal_lines:
            h_x1, h_y1, h_x2, h_y2 = h_line
            h_mid_x = (h_x1 + h_x2) / 2  # Midpoint x-coordinate of the horizontal line
            h_mid_y = (h_y1 + h_y2) / 2  # Midpoint y-coordinate of the horizontal line
            for v_line in vertical_lines:
                v_x1, v_y1, v_x2, v_y2 = v_line
                v_mid_x = (v_x1 + v_x2) / 2  # Midpoint x-coordinate of the vertical line
                v_mid_y = (v_y1 + v_y2) / 2  # Midpoint y-coordinate of the vertical line

                # Check proximity of each endpoint of the horizontal line to the vertical line's endpoints
                if (is_within_radius(v_x1, v_y1, h_x1, h_y1, radius) or
                    is_within_radius(v_x1, v_y1, h_x2, h_y2, radius) or
                    is_within_radius(v_x2, v_y2, h_x1, h_y1, radius) or
                    is_within_radius(v_x2, v_y2, h_x2, h_y2, radius)):

                    # Determine the direction of the turn based on the relative positions of line midpoints
                    if h_mid_x < v_mid_x and h_mid_y < v_mid_y:
                        turn_key = "right_turn"
                    elif h_mid_x > v_mid_x and h_mid_y < v_mid_y:
                        turn_key = "left_turn"
                    else:
                        continue  # Ignore if it does not meet the criteria for determining turn direction

                    lines_dict[turn_key] = [v_line, h_line]
                    
    if horizontal_lines:
        # Determine if any horizontal lines are touching the image borders
        # Potential future updates here: add a threshold for the length of the line touching the border (shorter lines should not be considiered because they could be generated based on obstacles)
        for h_line in horizontal_lines:
            h_x1, h_y1, h_x2, h_y2 = h_line
            h_mid_x = (h_x1 + h_x2) / 2  # Midpoint x-coordinate of the horizontal line
            h_mid_y = (h_y1 + h_y2) / 2  # Midpoint y-coordinate of the horizontal line

            if abs(h_x1) <= border_threshold:
                turn_key = "left_turn"  # Horizontal line touches the left border
            elif abs(image_width - h_x2) <= border_threshold:
                turn_key = "right_turn"  # Horizontal line touches the right border
            else:
                continue  # No border touch detected
            lines_dict[turn_key] = [h_line]

    # Post-process to ensure correct side classification
    for turn_type in ["left_turn", "right_turn"]:
        lines_to_keep = []
        for line in lines_dict.get(turn_type, []):
            v_line = line
            v_mid_x = (v_line[0] + v_line[2]) / 2
            if (turn_type == "left_turn" and v_mid_x < image_center_x) or (turn_type == "right_turn" and v_mid_x > image_center_x):
                lines_to_keep.append(line)
        lines_dict[turn_type] = lines_to_keep

    return lines_dict


def save_image(image_np, save_directory, filename):
    """
    Saves the provided NumPy array image to the specified directory with the given filename as a JPEG.
    
    Parameters:
    - image_np (numpy.ndarray): The image array to be saved, expected in HxWxC format.
    - save_directory (str): The directory where the image should be saved.
    - filename (str): The name of the file to save the image as, with .jpg extension.
    
    Returns:
    - str: The full path to the saved image.
    """
    
    # Ensure the save directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_np)
    
    # Construct the full path where the image will be saved
    save_path = os.path.join(save_directory, filename)
    
    # Save the image as JPEG
    image.save(save_path, 'JPEG')