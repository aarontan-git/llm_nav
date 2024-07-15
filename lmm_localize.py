import base64
import os, sys
import requests
import json
import argparse
import jsonlines
from datetime import datetime
from tqdm import tqdm
from tenacity import RetryError, retry, wait_random_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from helper import *
from vision_utils import *
import time
import argparse
import matplotlib.pyplot as plt
from gradio_client import Client, file
from instruction_parser import *
from process_hdmaps.generate_topomap import *
from combine_imgs import *
import shutil
from gpt_helper import *


API_KEY = "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"

"""
This script is used to try to build a LMM based localization method

run by:

python lmm_localize.py --data-file data/input_test.jsonl --mode llm_localize --parallel 1

"""

def encode_image(image_path):
    if not os.path.exists(image_path):
        print("not exist: ", image_path)
        exit(1)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(5))
def query_single_turn(image_paths, question, history=None, model="gpt-4-vision-preview", temperature=0, max_tokens=4096):
    '''
    Queries a single turn with the given parameters and returns the API response.

    Parameters:
    - image_paths (list of str): Paths to the images to be included in the query.
    - question (str): The text question to be asked.
    - history (list of dict, optional): Previous turns of the conversation. Defaults to None.
    - model (str): The model to be used for the query. Defaults to "gpt-4-vision-preview".
    - temperature (float): Controls randomness in the generation. Defaults to 0.
    - max_tokens (int): The maximum number of tokens to generate. Defaults to 4096.

    Returns:
    - dict: The JSON response from the API.
    '''
    
    content = [{"type": "text", "text": question}]
    for image_path in image_paths:
        encoded_image = encode_image(image_path)
        # i think the image gets encoded/uploaded here
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}})
    
    # # save content to json
    # with open(f"query_content_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w") as file:
    #     json.dump(content, file, indent=4)

    print("MODEL USED: ", model)

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    if history is not None:
        messages = history + messages
    
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url=url, headers=headers, json=payload)
    # print("response: ", response)

    response_json = response.json()
    # print("response json: ", response_json)
    response_text = response_json["choices"][0]["message"]["content"]
    
    # Regular expression to match each node entry
    pattern = r'\{"Node": (\d+), "Prob": ([\d.]+), "Landmarks": "(.*?)"\}'
    # Use regex findall to find all matches
    matches = re.findall(pattern, response_text, re.DOTALL)

    # Initialize dictionaries
    index_to_node = {}
    node_to_prob = {}
    node_to_landmarks = {}

    # Populate dictionaries
    for index, (node, prob, landmarks) in enumerate(matches):
        node_int = int(node)
        index_to_node[index] = node_int
        node_to_prob[node_int] = float(prob)
        node_to_landmarks[node_int] = landmarks

    # print("Index to Node:", index_to_node)
    # print("Node to Probability:", node_to_prob)
    # print("Node to Landmarks:", node_to_landmarks)

    response_info = [index_to_node, node_to_prob, node_to_landmarks]

    return response_json, response_info
    

def query_single_turn_and_save(exp_name, image_paths, question, system_msg=None, model="gpt-4-vision-preview", temperature=0, max_tokens=4096, additional_save=None):
    """
    Queries a single turn with the given parameters, saves the response along with additional experiment details to a JSON file.

    This function extends the functionality of `query_single_turn` by not only querying the model but also organizing the response and additional experiment details into a structured format and saving it to a file. This is particularly useful for logging and later analysis of the experiment results.

    Parameters:
    - exp_name (str): The name of the experiment. Used to name the saved file.
    - image_paths (list of str): Paths to the images to be included in the query.
    - question (str): The text question to be asked.
    - history (list of dict, optional): Previous turns of the conversation. Defaults to None.
    - model (str): The model to be used for the query. Defaults to "gpt-4-vision-preview".
    - temperature (float): Controls randomness in the generation. Defaults to 0.
    - max_tokens (int): The maximum number of tokens to generate. Defaults to 4096.
    - additional_save (dict, optional): Additional data to be saved along with the response. Defaults to None.
    """


    def get_today_str():
        # get today's date in format YYYY-MM-DD
        current_datetime = datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d")
        return date_string

    max_retries = 10
    attempts = 0
    response = None
    response_info = None

    while attempts < max_retries:
        try:
            response, response_info = query_single_turn(image_paths, question, system_msg, model, temperature, max_tokens)

            print("Response info: ", response_info)

            # Check if response_info[0] has at least one entry, indicating a successful response
            if response_info[0]:
                # If the query was successful and response_info[0] is not empty, break out of the loop
                break
            else:
                # If response_info[0] is empty, log the attempt and retry
                print(f">>> No nodes estimated on attempt {attempts + 1}, retrying...")
        except RetryError as e:
            print(f">>> API error on attempt {attempts + 1}: {e}")
        
        attempts += 1
        if attempts >= max_retries:
            print(">>> Max retries reached. Giving up.")
            break
        # Optional: Add a delay between retries if desired
        time.sleep(15)  # Sleep for 15 seconds before retrying

    # save the overall response
    overall = {
        "exp": exp_name,
        "image_paths": image_paths,
        "system_msg": system_msg,
        "question": question,
        "model": model,
        "temperature": temperature,
        "response": response
    }

    if additional_save is not None:
        overall.update(additional_save)
    save_dir = os.path.join("log", get_today_str())
    save_file = os.path.join(save_dir, f"{exp_name}.json")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        f.write(json.dumps(overall, indent=2, ensure_ascii=False))
        f.close()
    
    return response_info


def examples_inference(data_file="data/input.jsonl", mode="llm_localize", parallel=1, node_structure=None, model_name="gpt-4-vision-preview"):
    """
    Runs inference on examples from a JSONL data file using a specified model and mode, optionally in parallel.

    Parameters:
    - data_file (str): Path to the input JSONL file containing the examples.
    - mode (str): The mode of operation, e.g., 'llm_localize'.
    - parallel (int): Number of parallel queries to run. Defaults to 1 (sequential).
    - node_structure (str, optional): Textual description of the node structure. Defaults to None.
    - model_name (str): The model to be used for the query. Defaults to "gpt-4-vision-preview".

    Returns:
    - tuple: A tuple containing:
        - response_info (list): Information from the model's response.
        - system_prompt (list): The system prompt used in the query.
    """

    def get_textual_guidelines_single(node_structure):
        """
        this is currently used as the history (history only contains system message)
        """

        # Load the content of the text file
        with open('/home/asblab/aaron/s3/prompts/system_template.txt', 'r') as file:
            system_template = file.read()
        
        # Use f-string formatting to insert the node_structure into the template
        system_prompt = system_template.format(node_structure=node_structure)
        # print("formatted content: ", system_prompt)

        return [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": f"{system_prompt}"            
            }]
        }]
    
    def run_doc(doc, mode, node_structure):
        # Extract the image path from the document
        img_path = doc["image_path"]
        # Extract the question ID from the document
        id_ = doc["question_id"]
        # Format the experiment name using the mode and question ID
        exp_name = f"examples/{mode}/{id_}"
        # Extract the question text from the document
        qs = doc["question"]
        # Prepare additional data to be saved, in this case, the ground truth answer
        additional_save = {"ground_truth": doc["answer"]}
        # Check if the current mode is set to 'llm_localize'
        if mode == "llm_localize":
            # Assign the image path to a variable for clarity
            dots_image = img_path

            # Check if the processed image exists at the specified path
            if not os.path.exists(dots_image):
                # If the image does not exist, print an error message and return
                print(">>> processed image not exist:", dots_image, " please run image_processor.py first")
                return
            system_prompt = get_textual_guidelines_single(node_structure)
            # If the image exists, call the function to query the model and save the response
            response_info = query_single_turn_and_save(exp_name, [dots_image], qs, system_msg=system_prompt, model=model_name, additional_save=additional_save)
            return response_info, system_prompt
        else:
            # If an invalid mode is specified, print an error message and exit the program
            print(">>> invalid mode:", mode)
            exit(1)
            return None

    # Initialize an empty list to store documents
    docs = []
    # Open the data file using jsonlines to handle JSON objects line by line
    with jsonlines.open(data_file, "r") as f:
        # Iterate through each document in the file
        for doc in f:
            # Append each document to the docs list
            docs.append(doc)
        f.close()

    # save the docs to a json file
    with open("docs.json", "w") as f:
        f.write(json.dumps(docs, indent=2, ensure_ascii=False))
        f.close()

    # Check if the function is set to run in non-parallel mode (i.e., one document at a time)
    if parallel == 1:
        # Iterate through each document with a progress bar
        for doc in tqdm(docs):
            # Process each document according to the specified mode
            response_info, system_prompt = run_doc(doc, mode, node_structure)
    else:
        # If parallel processing is enabled, create a partial function with the mode set
        run_sample_and_save_wrapper = partial(run_doc, mode=mode)
        # Use ThreadPoolExecutor to run processes in parallel according to the specified number
        with ThreadPoolExecutor(parallel) as executor:
            # Map each document to the executor and process them in parallel, showing progress
            for _ in tqdm(
                executor.map(run_sample_and_save_wrapper, docs), total=len(docs)
            ):
                # Pass is used here as a placeholder since the action is performed by executor.map
                pass

    return response_info, system_prompt


def highlight_estimated_nodes(hdmap_np, hdmap_graph, estimated_nodes, save_path):
    """
    Highlights the estimated nodes on the hdmap_np image using shades of green to represent their estimated probabilities,
    and differentiates between numeric and non-numeric nodes by color. Numeric nodes not in the estimated list are shown in red,
    while non-numeric nodes are highlighted in brown. The function then saves the new image with these highlighted nodes.

    Parameters:
    - hdmap_np: The numpy array of the hand-drawn map image.
    - hdmap_graph: The graph object representing the topological map.
    - estimated_nodes: A list of estimated node IDs.

    The function saves the new image with highlighted nodes, ensuring numeric and non-numeric nodes are visually distinct.
    """

    # Ensure hdmap_np is of a compatible data type
    if hdmap_np.dtype not in [np.byte, np.short, np.float32, np.float64]:
        hdmap_np = hdmap_np.astype(np.float32)

    # Normalize hdmap_np to the range [0, 1] if it's a floating point type
    if np.issubdtype(hdmap_np.dtype, np.floating):
        hdmap_np = (hdmap_np - np.min(hdmap_np)) / (np.max(hdmap_np) - np.min(hdmap_np))

    # plt.figure(figsize=(10, 10))  # Set the figure size for better resolution
    plt.imshow(hdmap_np)  # Display the hand-drawn map

    pos = nx.get_node_attributes(hdmap_graph, 'pos')  # Get positions of all nodes

    # Draw all numeric nodes not in estimated_nodes in red
    numeric_nodes = list({node for node in hdmap_graph.nodes if str(node).isdigit()})
    non_numeric_nodes = set(hdmap_graph.nodes) - set(numeric_nodes)
    nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=set(numeric_nodes) - set(estimated_nodes), node_color='red', node_size=300)

    # Adjust the gradient generation to ensure it ranges from a fixed dark green to light green for the estimated nodes (to highlight the prediction)
    num_estimated_nodes = len(estimated_nodes)
    start_green = 0.95  # Lighter green, but not too light
    end_green = 0.35   # Darker green
    if num_estimated_nodes > 1: # to account for potential division by zero
        green_shades = [plt.cm.Greens(start_green - (start_green - end_green) * i / (num_estimated_nodes - 1)) for i in range(num_estimated_nodes)]
    else:
        green_shades = [plt.cm.Greens(start_green)]
    for i, node in enumerate(estimated_nodes):
        # Draw nodes in estimated_nodes with the corresponding shade of green
        nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=[node], node_color=[green_shades[i]], node_size=300)

    # draw the non-numeric nodes in brown (to highlight the landmarks) + labels
    for node in non_numeric_nodes:
        nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=[node], node_color='brown', node_shape='o', node_size=800)
    nx.draw_networkx_labels(hdmap_graph, pos, font_size=5, font_color='white', labels={node: node for node in non_numeric_nodes})

    # Draw all the edges in hdmap_graph (which at this point have already been pruned to only the necerssary edges)
    nx.draw_networkx_edges(hdmap_graph, pos, edge_color='black')

    # Draw labels for the numerical nodes
    nx.draw_networkx_labels(hdmap_graph, pos, font_size=10, font_color='white', labels={node: node for node in numeric_nodes})

    plt.axis('tight')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the plot to free up memory


def update_input_jsonl(data_file, image_path, previous_stuff, current_stuff, question):
    """
    Updates the `image_path` and `question` fields in the input JSONL file.

    Parameters:
    - data_file (str): Path to the input JSONL file.
    - image_path (str): Path to the image to be included in the query.
    - previous_stuff (str): Text representing previous estimations and actions.
    - current_stuff (str): Text representing current observations.
    - question (str): The text question to be asked.

    This function reads the input JSONL file, updates the `image_path` and `question` fields for each object,
    and writes the updated objects to a temporary file. The original file is then replaced with the updated file.
    """

    # Open the JSONL file for reading and writing
    with jsonlines.open(data_file, mode='r') as reader, jsonlines.open(data_file + ".tmp", mode='w') as writer:
        for obj in reader:
            # Update the image_path and question fields
            obj['image_path'] = image_path
            obj['question'] = previous_stuff + current_stuff + question
            # Write the updated object to the temporary file
            writer.write(obj)

    # Replace the original file with the updated temporary file
    os.replace(data_file + ".tmp", data_file)


def load_hdmap_data():
    """
    Loads the hand-drawn map (hdmap) numpy array and graph object from pickle files.

    Returns:
    - hdmap_np: Numpy array of the hand-drawn map image.
    - hdmap_graph: Graph object representing the topological map.
    """
    current_dir = os.path.dirname(__file__)
    hdmap_np_path = os.path.join(current_dir, 'process_hdmaps', 'quantized_image_np.pkl')
    hdmap_graph_path = os.path.join(current_dir, 'process_hdmaps', 'pruned_G.pkl')
    
    with open(hdmap_np_path, 'rb') as f:
        hdmap_np = pickle.load(f)
    
    with open(hdmap_graph_path, 'rb') as f:
        hdmap_graph = pickle.load(f)
    
    return hdmap_np, hdmap_graph, current_dir


def load_groundingdino_model():
    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    return load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)


def generate_node_structure(hdmap_graph, node_direction_map):
    """
    Generates a textual description of the node structure in the hdmap_graph.

    Parameters:
    - hdmap_graph: The graph object representing the topological map.
    - node_direction_map: A dictionary mapping nodes to their respective directions.

    Returns:
    - tuple: A tuple containing:
        - str: A string description of the node structure.
        - dict: A dictionary where each key is a node and the value is another dictionary with:
            - "Neighbors": A list of neighboring nodes.
            - "Landmarks": A string description of landmarks relative to the node.
    """
    node_structure = "Topological Map Structure: \n"
    node_structure_dict = {}
    # get the numeric nodes and sort them
    numeric_nodes = sorted([node for node in hdmap_graph.nodes if isinstance(node, int)])

    for node in numeric_nodes:
        if str(node).isdigit():
            digit_neighbors = [n for n in hdmap_graph.neighbors(node) if str(n).isdigit()]
            landmark_neighbors = [n for n in hdmap_graph.neighbors(node) if not str(n).isdigit()]
            position = hdmap_graph.nodes[node].get('pos', 'Position not available')
            direction = node_direction_map.get(int(node), 'N')  # Default to 'N' if direction is not found
            node_structure += f"Node {node} is connected to nodes: {digit_neighbors}.\n"
            
            landmark_descriptions = []
            for landmark in landmark_neighbors:
                # Get the position of the landmark node
                landmark_position = hdmap_graph.nodes[landmark].get('pos', 'Position not available')
                # Check if both the landmark position and the current node position are available
                if landmark_position != 'Position not available' and position != 'Position not available':
                    # Determine the direction of the landmark relative to the current node based on the node's direction
                    if direction == 'N':
                        direction_text = "on the right" if landmark_position[0] > position[0] else "on the left"
                    elif direction == 'S':
                        direction_text = "on the right" if landmark_position[0] < position[0] else "on the left"
                    elif direction == 'W':
                        direction_text = "on the right" if landmark_position[1] < position[1] else "on the left"
                    elif direction == 'E':
                        direction_text = "on the right" if landmark_position[1] > position[1] else "on the left"
                    # Append the landmark description to the list
                    landmark_descriptions.append(f"{landmark} {direction_text}")
                # Add the landmark descriptions to the node structure string
                node_structure += f"Node {node} has landmarks: {', '.join(landmark_descriptions)}.\n"
            
            node_structure_dict[node] = {
                "Neighbors": digit_neighbors,
                "Landmarks": ", ".join(landmark_descriptions)
            }
    
    return node_structure, node_structure_dict


def create_segments(nodes, shift_nodes):
    """
    Splits a list of nodes into segments based on specified shift nodes and duplicates each segment.

    Parameters:
    - nodes (list of int): A list of node IDs to be segmented.
    - shift_nodes (list of int): A list of node IDs that indicate where to split the segments.

    Returns:
    - list of list of int: A list of segments, where each segment is duplicated.
        - DUPLICATION ASSUMPTION IS MADE TO ALIGN THE OUTPUT SEGMENTS WITH THE LOCAL PLAN STRUCTURE.
    """
    segments = []
    current_segment = []
    shift_index = 0

    for node in nodes:
        current_segment.append(node)
        # Check if the current node matches the current shift node
        if shift_index < len(shift_nodes) and node == shift_nodes[shift_index]:
            segments.append(current_segment)
            segments.append(current_segment.copy())  # Duplicate the segment
            current_segment = []
            shift_index += 1

    # Add the last segment if it exists
    if current_segment:
        segments.append(current_segment)
        segments.append(current_segment.copy())  # Duplicate the segment

    return segments


def generate_expected_landmarks(node_structure_dict, local_node_segments):
    """
    Generates the expected landmarks from the given nodes without the direction.

    Parameters:
    - node_structure_dict (dict): Dictionary containing node information.
    - local_node_segments (list of int): List of node IDs to extract landmarks from.

    Returns:
    - str: A string of expected landmarks separated by " . ".
    """
    landmarks = []

    for node in local_node_segments:
        landmarks_str = node_structure_dict[node]['Landmarks']
        if landmarks_str:
            landmarks.extend([landmark.split(' ')[0] for landmark in landmarks_str.split(', ')])

    expected_landmarks = ' . '.join(landmarks)
    return expected_landmarks


def generate_local_node_structure(node_dict, local_node_segments):
    """
    Generates a textual description of the node structure for the given local node segments.

    Parameters:
    - node_dict (dict): Dictionary containing node information.
    - local_node_segments (list of int): List of node IDs to generate the structure for.

    Returns:
    - str: A string description of the node structure for the local node segments.
    """
    node_structure = ""
    
    for node in local_node_segments:
        neighbors = node_dict[node]['Neighbors']
        landmarks = node_dict[node]['Landmarks']
        
        node_structure += f"Node {node} is connected to nodes: {neighbors}.\n"
        if landmarks:
            node_structure += f"Node {node} has landmarks: {landmarks}.\n"
    
    return node_structure


def xget_available_actions(detection_result):
    """
    Determines the available actions based on the detection result.

    Parameters:
    - detection_result (list of dict): The detection result containing phrases and bounding boxes.

    Returns:
    - list of str: The list of available actions.
    """
    # print("detection result: ", detection_result)

    # Initialize the default action list
    action_list = ["GO_STRAIGHT", "STOP"] # should include GO BACK TO LAST CHECKPOINT

    # Check if "left_turn" or "right_turn" is in the detection result and add them to the action list
    for item in detection_result:
        if item["phrase"] == "left_turn" and "LEFT_TURN" not in action_list:
            action_list.append("LEFT_TURN")
        if item["phrase"] == "right_turn" and "RIGHT_TURN" not in action_list:
            action_list.append("RIGHT_TURN")

    print("Available Actions: ", action_list)
    return action_list


if __name__=="__main__":
    """
    This block is the entry point of the script when run as a standalone program.
    It parses command-line arguments, iterates over sorted indices to update and process input JSONL files,
    and performs image processing and inference tasks based on the provided arguments.
    """

    # INITIALIZATION ------------------------------------------------------------
    client = Client("https://b28018d8835b4c716a.gradio.live") # initialize the gradio client
    parser = argparse.ArgumentParser(description='LLM-Localize example.')
    parser.add_argument('--mode', type=str, help='Different prompting methods', default="llm_localize")
    parser.add_argument('--parallel', type=int, help='Number of parallel queries', default=1)
    parser.add_argument('--data-file', type=str, help='Data path')
    args = parser.parse_args()

    directory_path = "/home/asblab/aaron/files/output"
    sorted_indices = get_sorted_indices_from_directory(directory_path) # Use the function to get sorted indices
    hdmap_np, hdmap_graph, current_dir = load_hdmap_data() # load the pkl files for the hdmap_np, and hdmap_graph relative to the current file directory
    numeric_nodes = sorted([node for node in hdmap_graph.nodes if isinstance(node, int)], reverse=True)
    log_file_path = os.path.join(current_dir, "iteration_logs.txt") # Create the log file before the loop starts
    iteration_logger = IterationLogger(log_file_path)

    # INITIALIZE THE PROMPTS ----------------------------------------------------
    ## PREVIOUS STUFF
    with open('/home/asblab/aaron/s3/prompts/prev_estimation_template.txt', 'r') as f:
        prev_estimation_template = f.read()
    with open('/home/asblab/aaron/s3/prompts/prev_action_template.txt', 'r') as f:
        prev_action_template = f.read()
    ## CURRENT STUFF
    with open('/home/asblab/aaron/s3/prompts/current_view_description_template.txt', 'r') as f:
        current_view_description_template = f.read()
    with open('/home/asblab/aaron/s3/prompts/current_observation_template.txt', 'r') as f:
        current_observation_template = f.read()
    ## LOCALIZATION QUESTION
    with open('/home/asblab/aaron/s3/prompts/localization_question.txt', 'r') as f:
        localization_question = f.read()
    ## NAVIGATION QUESTION
    with open('/home/asblab/aaron/s3/prompts/navigation_question.txt', 'r') as f:
        navigation_template = f.read()

    # ------------------------------------------------------------------------

    # GENERATE THE NODE STRUCTURE IN TEXT FORM (NODE N IS CONNECTED TO NODES X, Y, Z. NODE N HAS LANDMARKS TO THE RIGHT/LEFT)
    node_direction_map = get_node_direction_map(hdmap_graph) # get the node direction map {node: direction}
    node_structure, node_structure_dict = generate_node_structure(hdmap_graph, node_direction_map)
    # print()
    # print("NODE STRUCTURE TIME --- ")
    # print(node_structure)
    # print()
    # print("NODE DICTIONARY TIME --- ")
    # print(node_structure_dict)

    # GENERATE THE GLOBAL NAVIGATION PLAN IN TEXT FORM (GO FROM NODE X TO NODE Y, THEN FROM NODE Y TO NODE Z)
    global_plan, shift_nodes = generate_global_navigation_plan(hdmap_graph, node_structure_dict)
    local_node_segments = create_segments(numeric_nodes, shift_nodes) # Generate the corresponding nodes for each segment of the local plan
    
    # Initialize the local plan status, plan index, image index, and time step
    local_plan_status = "incomplete"
    previous_landmark_description = ''


    plan_idx = 0
    image_idx = 0
    time_step = 0

    while local_plan_status != "complete":
        start_time = time.time()

        print()
        print("local plan: ", global_plan[plan_idx])
        print("local node segments: ", local_node_segments[plan_idx])
        print("time step: ", time_step)

        # RUNNING THE SPATIAL LANDMARK EXTRACTION API VIA GRADIO + GET LANDMARK DESCRIPTION -------------------------------------------------
        # Get the expected landmarks to look for (i.e., prompt for the open det/seg model)
        TEXT_PROMPT = generate_expected_landmarks(node_structure_dict, local_node_segments[plan_idx])

        # Get the API to extract landmarks from the robot POV
        parent_dir = os.path.dirname(current_dir)
        image_path = os.path.join(parent_dir, "files", "extracted_frames", f"frame_{image_idx}.jpg")
        print("image path: ", image_path)
        image_mask_tuple, labelled_image, detection_result = client.predict(
            image=file(image_path),
            text_prompt = TEXT_PROMPT,
            api_name="/predict"
        )
        # Save the labelled_image at the specified path for monitoring
        labelled_img_path = os.path.join(parent_dir, "files", "detection", f"frame_{image_idx}.jpg")
        shutil.copy(labelled_image, labelled_img_path) # Copy the labelled image from its temporary location to the specified path

        # classify the detected landmarks into quadrants to generate a text form description
        detection_result_quadrants = classify_boxes_by_quadrant(detection_result, 640, 480, middle_width_percent=20)
        landmark_description = filter_and_describe_objects(detection_result_quadrants)
        print("landmark_description: ", landmark_description)


        # SET THE ROBOT INITIAL POSE ----------------------------------------------------------------
        print()
        # print("Current progress: " + str(i) + "/" + str(len(sorted_indices)))
        if time_step == 0:
            # to initialize the process, we give the localization algo a starting point
            estimated_nodes = [9]
            node_to_prob = {9: 1.0}
            node_to_landmarks = {9: "The robot is at the starting node with the vending machine on the left."} # where?


        # GENERATE INPUT IMAGE BY COMBINING THE LOCAL TOPOMAP WITH THE ROBOT POV ------------------------------------------------
        hand_drawn_map_path = os.path.join(current_dir, "process_hdmaps", "local_topomap.png") # New local hand drawn map + topo graph
        frame_path = os.path.join(current_dir, "..", "files", "extracted_frames", f"frame_{image_idx}.jpg") # Current robot view
        combined_img_path = os.path.join(current_dir, "..", "files", "sensor_input", f"input_img_{image_idx}.png") # Combined image
        draw_graph_on_image(Image.fromarray(hdmap_np), hdmap_graph, hand_drawn_map_path, local_node_segments[plan_idx])
        combine_images(hand_drawn_map_path, frame_path, "Hand drawn map", "Front view", combined_img_path, compression_factor=0.4)


        # SET THE QUESTION TO BE ASKED TO THE LLM ----------------------------------------------------------------------------------
        prev_observation = '' # to be completed later
        prev_estimation = prev_estimation_template.format(estimated_nodes=estimated_nodes, node_to_prob=node_to_prob, node_to_landmarks=node_to_landmarks)
        prev_action = prev_action_template.format(last_action="MOVE_FORWARD")

        # use GPT to get a detailed description of the robot front view for the current observation
        current_view_description = asking_sam_altman(frame_path, current_view_description_template.format(landmark_description=landmark_description))
        current_observation = current_observation_template.format(view_description=current_view_description, landmark_description=landmark_description)
        
        # setting up the previous and current prompts for the llm
        previous_stuff = prev_estimation + prev_action + prev_observation
        current_stuff = current_observation
        question = localization_question
        
        # Update the input JSONL file for each index and add the text questions
        local_node_structure = generate_local_node_structure(node_structure_dict, local_node_segments[plan_idx])
        update_input_jsonl(data_file=args.data_file, image_path=combined_img_path, previous_stuff=prev_estimation, current_stuff=current_stuff, question=localization_question)
        response_info, system_prompt = examples_inference(data_file=args.data_file, mode=args.mode, parallel=args.parallel, node_structure=local_node_structure, model_name='gpt-4-turbo') # gets the llm estimation
        node_to_prob = response_info[1]
        node_to_landmarks = response_info[2]
        # get a list of nodes from estimated node dict
        estimated_nodes = [node for node in response_info[0].values()]
        print("estimated nodes: ", estimated_nodes)


        # TAKE THE LLM OUTPUT, AND ILLUSTRATE IT IN A FIGURE AND SAVE IT ---------------------------------------------------------
        save_path = os.path.join(current_dir, 'process_hdmaps', 'hdmap_pred_poses.png')
        highlight_estimated_nodes(hdmap_np, hdmap_graph, estimated_nodes, save_path) # save the estimated nodes to the hand drawn map
        hand_drawn_map = os.path.join(current_dir, "process_hdmaps", "hdmap_pred_poses.png")
        output_path = os.path.join(current_dir, "..", "files", "estimated_output", f"est_{image_idx}.jpg")
        combine_images(hand_drawn_map, frame_path, "Hand drawn map", "Front view", output_path, compression_factor=1, verbose=False)


        # Given the estimation output, ask llm to generate the action
        action_list = get_available_actions(detection_result)
        navigation_prompt = navigation_template.format(
            local_plan=global_plan[plan_idx],
            next_local_plan=global_plan[plan_idx+1],
            estimated_nodes=response_info,
            previous_action=global_plan[plan_idx], # placeholder for now
            previous_landmark_description=previous_landmark_description,
            current_landmark_description=landmark_description,
            action_list=action_list
        )
        
        
        action_output = asking_sam_altman(combined_img_path, navigation_prompt)
        print("action output: ", action_output)


        # LOG THE RESULTS --------------------------------------------------------------------------------------------
        complete_prompt = json.dumps(system_prompt) + "\n" + previous_stuff + "\n" + current_stuff + "\n" + question + "\n"
        iteration_logger.log_iteration(time_step, complete_prompt, response_info, navigation_prompt, action_output)
    
        print("Elapsed time: ", time.time() - start_time)
        print()


        # check to see when to move to the next plan NEEDS TO BE REVISED TO BE INTELLIGENT
        if time_step == 8:
            plan_idx+=1
        elif time_step == 9:
            plan_idx+=1
        elif time_step == 15:
            plan_idx+=1
        elif time_step == 16:
            plan_idx+=1
        elif time_step == 17:
            local_plan_status = "complete"

        image_idx +=1
        time_step +=1

        previous_landmark_description = landmark_description


