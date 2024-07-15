import os
from gradio_client import Client, file
from PIL import Image

client = Client("https://d9ba28985c1337716e.gradio.live")

# Generate file names based on the given list of indices
indices = [45, 450, 900, 1350, 1800, 2400, 2850, 3600, 4275, 4950, 5700, 6300, 6750, 7500, 8150]  # Example list of indices
# Example dictionary of image indices and their corresponding text prompts
text_prompts = {
    45: ["TV . elevator door"],
    450: ["garbage bin . TV . poster"],
    900: ["poster"],
    1350: ["garbage . garbage bin"],
    1800: ["stairs . garbage bin"],
    2400: ["desks . tool box . ladder"],
    2850: ["plant . desks"],
    3600: ["plant"],
    4275: ["garbage bin . desks . plant"],
    4950: ["TV . plant"],
    5700: ["garbage bin . desks"],
    6300: ["garbage bin . desks"],
    6750: ["door . bench"],
    7500: ["door . bench"],
    8150: ["flat screen TV . bench"],
}

file_names = [f"image_{idx}.png" for idx in indices]
directory = "/home/asblab/aaron/files/hardware_trials/trial_2_images"
output_directory = "/home/asblab/aaron/files/hardware_trials/labelled_img_turns"  # Specify your output directory here

def process_single_image(file_name):
    file_path = os.path.join(directory, file_name)

    print(f"Processing file: {file_name}")

    if os.path.exists(file_path):
        # Extract the index from the filename
        image_idx = int(file_name.split('_')[1].split('.')[0])
        text_prompt = text_prompts.get(image_idx, ["default prompt"])[0]

        result, labelled_image, detection_result = client.predict(
            image=file(file_path),
            text_prompt=text_prompt,
            api_name="/predict"
        )
    
        # Save the labelled image as a PNG file
        img = Image.open(labelled_image)
        output_path = os.path.join(output_directory, file_name)
        img.save(output_path, "PNG")


def main():
    for file_name in file_names:
        process_single_image(file_name)
    
    # process_single_image("image_4275.png")


if __name__ == "__main__":
    main()