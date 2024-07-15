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
    nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=set(numeric_nodes) - set(estimated_nodes), node_color='orange', node_size=200)

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
        nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=[node], node_color=[green_shades[i]], node_size=200)

    # draw the non-numeric nodes in brown (to highlight the landmarks) + labels
    for node in non_numeric_nodes:
        nx.draw_networkx_nodes(hdmap_graph, pos, nodelist=[node], node_color='brown', node_shape='o', node_size=800)
    nx.draw_networkx_labels(hdmap_graph, pos, font_size=5, font_color='white', labels={node: node for node in non_numeric_nodes})

    # Draw all the edges in hdmap_graph (which at this point have already been pruned to only the necerssary edges)
    # nx.draw_networkx_edges(hdmap_graph, pos, edge_color='black')

    # Draw labels for the numerical nodes
    nx.draw_networkx_labels(hdmap_graph, pos, font_size=7, font_color='black', labels={node: node for node in numeric_nodes})

    plt.axis('tight')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the plot to free up memory




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



if __name__=="__main__":
    """
    This block is the entry point of the script when run as a standalone program.
    It parses command-line arguments, iterates over sorted indices to update and process input JSONL files,
    and performs image processing and inference tasks based on the provided arguments.
    """

    # INITIALIZATION ------------------------------------------------------------
    hdmap_np, hdmap_graph, current_dir = load_hdmap_data() # load the pkl files for the hdmap_np, and hdmap_graph relative to the current file directory
    numeric_nodes = sorted([node for node in hdmap_graph.nodes if isinstance(node, int)], reverse=True)

    # TAKE THE LLM OUTPUT, AND ILLUSTRATE IT IN A FIGURE AND SAVE IT ---------------------------------------------------------
    estimated_nodes_list = [
        [0, 2, 1],
        [0, 1, 2],
        [2, 3, 1],
        [3, 4, 2, 5],
        [4, 5, 7],
        [6, 4, 8, 10],
        [8, 6, 11],
        [11, 16, 8],
        [23, 19, 25, 16],
        [27, 29, 25, 26],
        [32, 29, 31, 27],
        [32, 31, 29, 34],
        [34, 37, 32, 35],
        [38, 36, 37],
        [36, 38]
    ]
    save_directory = os.path.join(current_dir, 'process_hdmaps/hw_estimates')

    for idx, estimated_nodes in enumerate(estimated_nodes_list):
        save_path = os.path.join(save_directory, f'estimate_{idx}.jpg')
        highlight_estimated_nodes(hdmap_np, hdmap_graph, estimated_nodes, save_path) # save the estimated nodes to the hand drawn map
