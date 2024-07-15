import os
from segment_anything import build_sam, SamPredictor 
from vision_utils import *
from lmm_localize import *
from open_det_seg import *
import gradio as gr

# Load models outside the function to avoid reloading them on each call
groundingdino_model = load_groundingdino_model()
sam_checkpoint = '/home/asblab/aaron/files/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device='cuda:0')
sam_predictor = SamPredictor(sam)

def landmark_extraction(image, text_prompt):
    """
    Processes an input image to extract and label spatial landmarks based on predefined categories.
    The function saves the uploaded image, performs object detection and semantic segmentation,
    and then categorizes and labels detected lines in the image as various types of turns or slopes.

    Parameters:
    - image (numpy.ndarray): The input image in NumPy array format.
    - text_prompt (str): The text prompt to guide the detection and segmentation.

    Returns:
    - tuple: A tuple containing:
        - numpy.ndarray: The processed image with landmarks labeled.
        - list: A list of tuples where each tuple contains a mask array and its corresponding label (mask_array, label_str).
    - numpy.ndarray: The labelled image with lines and turns annotated.
    - list: A list of dictionaries with detection results.
    """

    # Save the user uploaded image
    image_dir = '/home/asblab/aaron/s3/segmented'
    filename = 'processed_image.jpg'
    print(f"SAVING RETRIEVED IMAGE AT {image_dir} ...")
    save_image(image, image_dir, filename)

    TEXT_PROMPT = 'floor . ' + text_prompt # It is always looking for floor and then the user prompt
    save_directory = '/home/asblab/aaron/s3/segmented'  # Directory to save the output

    # Run open vocabulary detection and process the output
    detection_output, image_output = open_det_inference(image_dir, filename, groundingdino_model, TEXT_PROMPT, save_directory)
    image_source = image_output[0]
    annotated_detection_frame = image_output[1]
    image = image_output[2]
    boxes = detection_output[0]
    logits = detection_output[1]
    phrases = detection_output[2]

    if phrases != []:
        # Run open vocabulary segmenetation and process the output
        annotated_segmentation_frame, seg_masks = open_seg_inference(sam_predictor, image_source, filename, annotated_detection_frame, boxes, save_directory)
        # save_mask_dir = '/home/asblab/aaron/s3/open_api/mask'
        masks_images_np = process_masks(seg_masks) # convert seg_masks (bool tensor) to np array
        masks = create_unique_keyed_dict(phrases, masks_images_np) # generate a list of masks [(mask_array, label), (mask_array, label), ...]
        mask_output = [(data, label) for label, data in masks.items()]
        detection_output[2] = list(masks.keys()) # Update detection_output phrases with unique mask names

        # Extract left/right turn from image
        lines = get_edges(image_source, mask_output, filename)
        # Change the threshold for catgorize_lines to adjust the sensitivity of the turn detection
        line_categories_dict = categorize_lines(lines, threshold=5, image_height=image.height, min_line_length=10) # get line categories: positive_slope, negative_slope, vertical, horizontal
        line_categories_dict = find_turns(image_dir, filename, line_categories_dict, radius=50) # based on the categories, add 2 new categories: left_turn, right_turn
        labelled_image, turn_masks, turn_bounding_boxes = draw_lines_on_image(image_dir, filename, line_categories_dict, save_directory, ['positive_slope', 'negative_slope', 'vertical', 'horizontal', 'left_turn', 'right_turn'])        
        mask_output.extend(turn_masks) # add turn_masks to mask_output

        # Add turn bounding boxes to detection output
        for turn_type, turn_boxes in turn_bounding_boxes.items():
            for box in turn_boxes:
                detection_output[0] = torch.cat((detection_output[0], box.unsqueeze(0)), dim=0)
                detection_output[1] = torch.cat((detection_output[1], torch.tensor([1.0])), dim=0)
                detection_output[2].append(turn_type)

        # label the image with left/right turn results
        labelled_image, _ = annotate(image_source=labelled_image, boxes=detection_output[0], logits=detection_output[1], phrases=detection_output[2])

        # generate detection results in json format for gradio
        detection_results = det_json(detection_output)

    else: # if nothing is detected
        # generate a blank mask_output
        mask_output = [(np.zeros(image.size[::-1]), 'None')]
        detection_results = [{'phrase': 'None', 'box': [0, 0, 0, 0], 'logit': 0}]

    return (image, mask_output), labelled_image, detection_results


def main():
    # Set up the Gradio interface for API
    mask_segments = gr.AnnotatedImage()
    demo = gr.Interface(
        fn=landmark_extraction,
        inputs=[gr.Image(label="Image"), gr.Textbox(label="Enter Text Prompt (i.e., table . chair . person)")],
        outputs=[mask_segments, "image", "json"]
    )
    demo.launch(share=True)

if __name__ == "__main__":
    main()