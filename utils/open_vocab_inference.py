import os, sys
import os
import re
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import supervision as sv

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
from huggingface_hub import hf_hub_download


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
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

    return image_source, annotated_frame, boxes


def make_predictions_on_folder(input_folder, text_prompt, save_directory='/home/asblab/aaron/files/detection/'):
    """
    Runs detection on each image in the specified folder with the given text prompt and saves the output images to another directory.

    Parameters:
    - input_folder (str): Path to the folder containing image files.
    - text_prompt (str): Text prompt for the predictions.
    - save_directory (str, optional): Directory to save the output images. Defaults to '/home/asblab/aaron/files/detection/'.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Function to extract numerical part of the filename
    def extract_number(filename):
        s = re.findall("\d+", filename)
        return int(s[0]) if s else -1

    # Sort files by their numerical index
    sorted_filenames = sorted(os.listdir(input_folder), key=extract_number)

    # Iterate through sorted files in the input folder
    for filename in sorted_filenames:
        # Construct the full file path
        file_path = os.path.join(input_folder, filename)
        # Check if the current file is an image (for simplicity, checking by extension)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            print(f"Processing {filename}...")
            try:
                # Make a single prediction on the image
                make_single_prediction(image_path=file_path, text_prompt=text_prompt, save_directory=save_directory)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


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


def segment_and_save_with_masks(image_source, filename, annotated_image, boxes, save_directory):
    """
    Segments the given image based on the provided boxes, overlays the segmentation masks on the annotated image,
    and saves the result with a filename that includes the mask index matching the input file name.

    Parameters:
    - image_source (numpy.ndarray): The source image array.
    - annotated_image (numpy.ndarray): The image array with annotations.
    - boxes (torch.Tensor): The bounding boxes for segmentation.
    - save_directory (str): Directory to save the output image with masks.
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
    print(f"Annotated image with masks saved at {output_image_path}")


if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
    # If you have multiple GPUs, you can set the GPU to use here.
    # The default is to use the first GPU, which is usually GPU 0.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    mode = 'seg'  # 'det' for detection, 'seg' for segmentation
    process_type = 'single_image'  # 'single_image' or 'folder'
    image_path = '/home/asblab/aaron/files/YC_Videos/Act_1/frame_0.jpg'  # Path to the image or folder
    TEXT_PROMPT = "door . desk . person . plant . bench . chair . poster . turtlebot . lights . garbage bin . window . floor . ladder . laptop . tripod . stairs . rails"
    save_directory_det = '/home/asblab/aaron/files/YC_Videos/Act_1_Det'
    save_directory_seg = '/home/asblab/aaron/files/YC_Videos/Act_1_Seg'


    sam_checkpoint = '/home/asblab/aaron/files/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device='cuda:0')
    sam_predictor = SamPredictor(sam)


    # image_source, annotated_image, boxes = make_single_prediction(image_path=image_path, text_prompt=TEXT_PROMPT, save_directory=save_directory_det)
    # segment_and_save_with_masks(image_source=image_source, annotated_image=annotated_image, boxes=boxes, save_directory=save_directory_seg)


    if process_type == 'single_image':
        if mode == 'det':
            make_single_prediction(image_path=image_path, text_prompt=TEXT_PROMPT, save_directory=save_directory_det)
        elif mode == 'seg':
            image_source, annotated_image, boxes = make_single_prediction(image_path=image_path, text_prompt=TEXT_PROMPT, save_directory=save_directory_det)
            
            filename = os.path.basename(image_path)
            
            segment_and_save_with_masks(image_source=image_source, filename=filename, annotated_image=annotated_image, boxes=boxes, save_directory=save_directory_seg)
    elif process_type == 'folder':
        if mode == 'det':
            make_predictions_on_folder(input_folder=image_path, text_prompt=TEXT_PROMPT, save_directory=save_directory_det)
        elif mode == 'seg':
            # Ensure the save directory exists
            if not os.path.exists(save_directory_seg):
                os.makedirs(save_directory_seg)

            # Function to extract numerical part of the filename
            def extract_number(filename):
                s = re.findall("\d+", filename)
                return int(s[0]) if s else -1

            # Sort files by their numerical index
            sorted_filenames = sorted(os.listdir(image_path), key=extract_number)

            # Iterate through sorted files in the input folder
            for filename in sorted_filenames:

                cur_image_index = sorted_filenames.index(filename) + 1
                total_images = len(sorted_filenames)
                print()
                print(f"Processing image {cur_image_index}/{total_images}...")

                # Construct the full file path
                file_path = os.path.join(image_path, filename)

                print("FILENAME: ", filename)

                # Check if the current file is an image (for simplicity, checking by extension)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    print(f"Processing {filename}...")
                    try:
                        # Make a single prediction on the image
                        image_source, annotated_image, boxes = make_single_prediction(image_path=file_path, text_prompt=TEXT_PROMPT, save_directory=save_directory_det)
                        # Segment and save with masks
                        segment_and_save_with_masks(image_source=image_source, filename = filename, annotated_image=annotated_image, boxes=boxes, save_directory=save_directory_seg)
                    except Exception as e:
                        print(f"Failed to process {filename}: {e}")