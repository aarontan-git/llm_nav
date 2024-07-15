import gradio as gr
from vision_utils import *
from lmm_localize import *
import torchvision

# Load models outside the function to avoid reloading them on each call
groundingdino_model = load_groundingdino_model()
sam_checkpoint = '/home/asblab/aaron/files/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device='cuda:0')
sam_predictor = SamPredictor(sam)

def det_json(detection_output):
    """
    Converts detection output into a JSON-serializable list of dictionaries.

    Each dictionary contains a bounding box, a logit score, and a phrase associated with the detected object.

    Args:
        detection_output (tuple): A tuple containing three elements:
            - boxes_tensor (torch.Tensor): Tensor of bounding boxes with shape [n, 4].
            - logits (torch.Tensor): Tensor of logit scores with shape [n].
            - phrases (list of str): List of phrases corresponding to each detected object.

    Returns:
        list of dict: A list where each dictionary contains 'phrase', 'box', and 'logit' keys.
    """
    boxes_tensor = detection_output[0]  # Tensor of bounding boxes
    logits = detection_output[1]        # Tensor of logit scores
    phrases = detection_output[2]       # List of phrases associated with each detection

    # Create a list of dictionaries, each containing the box, corresponding logit, and phrase
    results = []
    for box, logit, phrase in zip(boxes_tensor, logits, phrases):
        results.append({
            "phrase": phrase,  # Already a string
            "box": box.tolist(),  # Convert tensor to list for JSON serialization
            "logit": logit.item()  # Convert tensor to Python scalar
        })

    return results


def get_unique_filename(directory, base_filename):
    """
    Generates a unique filename within the specified directory by appending a counter to the base filename if needed.

    Args:
    directory (str): The directory in which to check for uniqueness.
    base_filename (str): The initial filename to use as a base for generating a unique filename.

    Returns:
    str: A unique filename within the specified directory.

    This function checks if the file exists in the given directory and if it does, it appends an incrementing number
    to the base filename until a unique filename is found. This ensures no existing file is overwritten.
    """
    name, ext = os.path.splitext(base_filename)
    counter = 1
    unique_filename = base_filename
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{name}_{counter}{ext}"
        counter += 1
    return unique_filename


def process_masks(seg_masks, save_directory = None):
    """
    Processes segmentation masks by converting them from bool tensor to binary images, saving them as JPEG files,
    and storing their numpy array representations in a list.

    Args:
    seg_masks (torch.Tensor): A bool tensor containing segmentation masks with dimensions [n, 1, h, w],
                              where n is the number of masks, 1 is the channel dimension, and h, w are the height and width.
    save_directory (str): The directory path where the mask images will be saved.

    Returns:
    list: A list of numpy arrays, each representing a binary mask image.
    """
    mask_images_np = []  # Initialize list to store numpy arrays of masks
    for i, mask in enumerate(seg_masks):
        mask_image = mask.squeeze()  # Remove channel dimension if it's 1
        mask_image = (mask_image > 0).float()  # Convert to binary image
        mask_image_np = mask_image.cpu().numpy()  # Convert tensor to numpy array
        mask_image_np = mask_image_np.astype(int)
        mask_images_np.append(mask_image_np)  # Add numpy array to list
        if save_directory != None:
            filename = f"mask_{i}.jpg"
            full_path = os.path.join(save_directory, filename)
            torchvision.utils.save_image(mask_image, full_path)
    return mask_images_np


def create_unique_keyed_dict(phrases, masks_images_np):
    """
    Creates a dictionary where each key is a unique phrase associated with a mask image.

    This function ensures that if the same phrase appears multiple times in the input list,
    each occurrence is made unique by appending an incrementing number to the phrase.

    Args:
    phrases (list of str): List of phrases, where each phrase corresponds to a mask.
    masks_images_np (list of numpy arrays): List of numpy arrays, each representing a mask image.

    Returns:
    dict: A dictionary with unique phrases as keys and corresponding mask images as values.
    """
    masks = {}
    phrase_count = {}
    for phrase, mask in zip(phrases, masks_images_np):
        if phrase in phrase_count:
            # Increment the count for this phrase
            phrase_count[phrase] += 1
            # Append the count to the phrase to make it unique
            unique_phrase = f"{phrase}_{phrase_count[phrase]}"
        else:
            # Initialize the count for this phrase
            phrase_count[phrase] = 1
            unique_phrase = phrase
        
        # Add to dictionary with the unique phrase
        masks[unique_phrase] = mask
    return masks


def open_det_seg_inference(image, text_prompt):
    # Save the user uploaded image
    image_dir = '/home/asblab/aaron/s3/open_api/uploaded'
    base_filename = 'uploaded_image.jpg'
    filename = get_unique_filename(image_dir, base_filename)
    save_image(image, image_dir, filename)

    # Start the progress to generate the labelled image
    TEXT_PROMPT = text_prompt
    save_directory = '/home/asblab/aaron/s3/open_api/uploaded'  # Directory to save the output

    # Run open vocabulary detection and process the output
    detection_output, image_output = open_det_inference(image_dir, filename, groundingdino_model, TEXT_PROMPT, save_directory)
    image_source = image_output[0]
    annotated_detection_frame = image_output[1]
    image = image_output[2]
    boxes = detection_output[0]
    phrases = detection_output[2]

    if phrases != []:
        # Run open vocabulary segmenetation and process the output
        annotated_segmentation_frame, seg_masks = open_seg_inference(sam_predictor, image_source, filename, annotated_detection_frame, boxes, save_directory)
        save_mask_dir = '/home/asblab/aaron/s3/open_api/mask'
        masks_images_np = process_masks(seg_masks, save_mask_dir) # convert seg_masks (bool tensor) to np array
        masks = create_unique_keyed_dict(phrases, masks_images_np) # generate a list of masks [(mask_array, label), (mask_array, label), ...]
        mask_output = [(data, label) for label, data in masks.items()]

        # convert detection_output to json
        # print("detection output: ", detection_output)
        detection_results = det_json(detection_output)

    else: # if nothing is detected
        # generate a blank mask_output
        mask_output = [(np.zeros(image.size[::-1]), 'None')]
        detection_results = [{'phrase': 'None', 'box': [0, 0, 0, 0], 'logit': 0}]

    return (image, mask_output), detection_results


def main():
    # Set up the Gradio interface for API
    img_section = gr.AnnotatedImage()
    demo = gr.Interface(
        fn=open_det_seg_inference,
        inputs=[gr.Image(label="Image"), gr.Textbox(label="Enter Text Prompt (i.e., floor . table . chair)")],
        outputs=[img_section, "json"]
    )
    demo.launch(share=True, show_error=True)


if __name__ == "__main__":
    main()