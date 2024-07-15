from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
from skimage.draw import line
import pickle

# Authentication to Google API
import io
import os
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/asblab/aaron/s3/process_hdmaps/meta-coyote-419915-054216820d79.json'
from PIL import Image, ImageDraw

def quantize_image(image_np, n_colors):
    """
    This function quantizes an image into a specified number of colors using KMeans clustering.
    It reduces the color palette of the image to 'n_colors' colors, which can be useful for image compression
    or simplifying the image for further processing.

    Parameters:
    - image_np: A numpy array of the image to be quantized. Expected to be in the format (height, width, 3) for RGB images.
    - n_colors: The number of colors to reduce the image to.

    Returns:
    - quantized_image: The quantized image as a numpy array of the same shape as 'image_np' but with reduced color palette.
    """

    # Reshaping the image into a 2D array where each row represents a pixel and the three columns represent the RGB values.
    # This is necessary because KMeans clustering expects data in this format.
    pixels = image_np.reshape(-1, 3)
    
    # Performing KMeans clustering on the reshaped image data. 'n_clusters' is set to the desired number of colors.
    # The fit method finds the best 'n_colors' clusters of pixels in the color space of the image.
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    
    # Replacing each pixel's color with the color of the cluster center it belongs to.
    # 'kmeans.labels_' gives the cluster assignment for each pixel, and 'kmeans.cluster_centers_' gives the color of each cluster center.
    # This step effectively reduces the image's color palette to 'n_colors' colors.
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshaping the quantized pixels back to the original image shape to get the quantized image.
    # The data type is set to 'uint8' to represent pixel values as integers between 0 and 255.
    quantized_image = new_pixels.reshape(image_np.shape).astype('uint8')
    
    return quantized_image


def create_indexed_color_image_at_resolution(quantized_image_np, target_width):
    """
    Create an indexed color image at a user specified resolution, maintaining the aspect ratio.
    
    Parameters:
    - quantized_image_np: Numpy array of the quantized image.
    - target_width: The target width for the output image.
    
    Returns:
    - indexed_color_image_resized: The indexed color image at the specified resolution.
    """
    from skimage.transform import resize
    
    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = quantized_image_np.shape[0] / quantized_image_np.shape[1]
    target_height = int(target_width * aspect_ratio)
    
    # Resize the quantized image to the target resolution
    quantized_image_resized = resize(quantized_image_np, (target_height, target_width), 
                                      order=0, preserve_range=True, anti_aliasing=False).astype(int)
    
    # Calculate unique colors in the resized image
    unique_colors_resized, counts_resized = np.unique(quantized_image_resized.reshape(-1, 3), axis=0, return_counts=True)
    
    # Create a 2D array to hold the index of the color for each pixel in the resized image
    indexed_color_image_resized = np.zeros((target_height, target_width), dtype=int)
    
    # Iterate over unique colors to create a mapping from color to index for the resized image
    color_to_index_resized = {tuple(color): idx for idx, color in enumerate(unique_colors_resized)}
    
    # Populate the indexed_color_image_resized with the index of each pixel's color
    for i in range(target_height):
        for j in range(target_width):
            indexed_color_image_resized[i, j] = color_to_index_resized[tuple(quantized_image_resized[i, j])]
    
    return indexed_color_image_resized


# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# Function to draw a line and return pixel values along the line
def get_line_pixel_values(img, start, end):
    # Use skimage's line function to get pixel coordinates
    rr, cc = line(start[0], start[1], end[0], end[1])  # Note: line expects (y, x) format for start and end
    # Sample the pixel values along the line
    pixel_values = img[rr, cc]
    return pixel_values


def find_and_plot_centroids(image, cell_value, n_clusters):
    """
    Find centroids of clusters formed by cells with a specific value in an image,
    plot these centroids on the image, and save the figure.

    Parameters:
    - image: Numpy array of the indexed color image.
    - cell_value: The cell value to cluster.
    - n_clusters: Number of clusters to form.
    """
    # Step 1: Extract coordinates of cells with the specified value
    coords = np.column_stack(np.where(image == cell_value))

    # Step 2: Use KMeans to cluster these coordinates
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_

    # Step 3: Round centroids to nearest integer to use as indices (2D pixel values)
    centroids = np.rint(centroids).astype(int)

    return centroids


def create_fully_connected_graph(centroids):
    """
    Create a fully connected graph from a list of centroids.

    Parameters:
    - centroids: A list of centroids where each centroid is a tuple or list of (y, x) coordinates.

    Returns:
    - G: A fully connected networkx graph with nodes positioned at the centroids.
    """
    G = nx.Graph()

    # Add nodes with centroids positions
    for i, centroid in enumerate(centroids):
        G.add_node(i, pos=(centroid[1], centroid[0]))


    # Add nodes with centroids positions
    for i, centroid in enumerate(centroids):
        # Note: centroid is expected in (y, x) format, but pos is stored as (x, y) for consistency with plotting
        G.add_node(i, pos=(centroid[1], centroid[0]))

    # Connect each node to every other node (fully connected graph)
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')

    return G, pos


def calculate_max_distance_for_edges(G, pos, weight_multiplier=10, default_distance=100):
    """
    Calculate the maximum distance to set for edges in a graph based on the average of the shortest distances
    between each node and its nearest neighbor, adjusted by a weight multiplier.

    Parameters:
    - G: The graph (networkx.Graph).
    - pos: Dictionary of node positions {node: (x, y)}.
    - weight_multiplier: Multiplier to adjust the average shortest distance (default: 1.6).
    - default_distance: Default max distance in case there are no distances (default: 100).

    Returns:
    - max_distance: The calculated maximum distance to use for setting edges.
    """
    shortest_distances = []

    # Iterate through each node in the graph
    for node in G.nodes:
        node_pos = pos[node]
        distances = []
        # Calculate distance from this node to all other nodes
        for neighbor in G.nodes:
            if node != neighbor:
                neighbor_pos = pos[neighbor]
                distance = calculate_distance(node_pos, neighbor_pos)
                distances.append(distance)
        # Find the shortest distance for this node
        if distances:
            shortest_distances.append(min(distances))

    # Calculate the average of these shortest distances
    if shortest_distances:
        average_shortest_distance = np.mean(shortest_distances)
        max_distance = average_shortest_distance * weight_multiplier
    else:
        max_distance = default_distance  # Default value in case there are no distances

    return max_distance


def prune_graph_edges(G, max_distance, threshold_percentage, pos, color_image):
    """
    Prune edges from the graph based on two criteria:
    1. Remove edges where the percentage of a specific value (2) along the line between nodes is below a threshold.
    2. Remove edges that exceed a specified maximum distance.

    Parameters:
    - G: The input graph to be pruned.
    - max_distance: The maximum allowed distance between nodes for an edge to be retained.
    - threshold_percentage: The minimum percentage of a specific value (2) required along the line between nodes.
    - pos: Dictionary of node positions {node: (x, y)}.
    - indexed_color_image_resized: The indexed color image resized to the target width.

    Returns:
    - G_pruned: The pruned graph.
    """
    # Create a copy of the graph to modify
    G_pruned = G.copy()

    # First, prune based on the threshold percentage
    for edge in G_pruned.edges():
        node, neighbor = edge
        node_pos = pos[node]
        neighbor_pos = pos[neighbor]

        # Convert positions from (x, y) to (y, x) because image arrays are indexed as [row, column]
        node_pos_yx = (node_pos[1], node_pos[0])
        neighbor_pos_yx = (neighbor_pos[1], neighbor_pos[0])


        # Get pixel values along the line between node and neighbor
        line_pixel_values = get_line_pixel_values(color_image, node_pos_yx, neighbor_pos_yx)

        # Calculate unique values and their counts
        unique_values, counts = np.unique(line_pixel_values, return_counts=True)

        # # Check if the value 2 meets the threshold
        # if 3 in unique_values:
        #     index_of_2 = np.where(unique_values == 3)[0][0]
        #     percentage_of_2 = (counts[index_of_2] / sum(counts)) * 100
        #     if percentage_of_2 < threshold_percentage:
        #         # Remove the edge if it doesn't meet the threshold
        #         G_pruned.remove_edge(node, neighbor)
        
        # Determine the index for the last unique value, which is the target color for pruning
        target_color_index = len(unique_values) - 1  # Index of the last unique value
        target_color = unique_values[target_color_index]

        # Calculate the percentage of the target color along the line
        if target_color in unique_values:
            index_of_target_color = np.where(unique_values == target_color)[0][0]
            percentage_of_target_color = (counts[index_of_target_color] / sum(counts)) * 100
            if percentage_of_target_color < threshold_percentage:
                # Remove the edge if it doesn't meet the threshold
                G_pruned.remove_edge(node, neighbor)
        else:
            # Remove the edge if 2 is not among the unique values
            G_pruned.remove_edge(node, neighbor)

    # Then, prune based on the maximum distance
    for edge in list(G_pruned.edges()):
        node, neighbor = edge
        node_pos = pos[node]
        neighbor_pos = pos[neighbor]

        # Convert positions from (x, y) to (y, x) for distance calculation
        node_pos_yx = (node_pos[1], node_pos[0])
        neighbor_pos_yx = (neighbor_pos[1], neighbor_pos[0])

        # Calculate the distance between the node and its neighbor
        distance = calculate_distance(node_pos_yx, neighbor_pos_yx)

        # Check if the distance exceeds the maximum allowed distance
        if distance > max_distance:
            G_pruned.remove_edge(node, neighbor)

    return G_pruned


def draw_bounding_boxes(hdmap_np, text_annotations):
    """
    This function annotates the hdmap_np image by drawing bounding boxes around text identified by the Google Vision API.

    Args:
    hdmap_np (numpy.ndarray): A numpy array representing the hand-drawn map image.
    text_annotations (list): A list of text annotations provided by the Google Vision API.

    Returns:
    numpy.ndarray: The annotated hdmap_np image with bounding boxes.
    """
    import cv2
    import numpy as np

    # Convert hdmap_np to a format that can be used with OpenCV if it's not already
    if hdmap_np.dtype != np.uint8:
        hdmap_np = (hdmap_np * 255).astype(np.uint8)

    # Ensure hdmap_np is in color (3 channels)
    if len(hdmap_np.shape) == 2:  # if it's grayscale
        hdmap_np = cv2.cvtColor(hdmap_np, cv2.COLOR_GRAY2BGR)

    # Skip the first text_annotation since it contains a summary of all detected text
    for annotation in text_annotations[1:]:  # Start from the second element
        vertices = annotation.bounding_poly.vertices
        # Find the min and max x and y coordinates to ensure proper rectangle dimensions
        xs = [vertex.x for vertex in vertices]
        ys = [vertex.y for vertex in vertices]
        top_left = (min(xs), min(ys))
        bottom_right = (max(xs), max(ys))
        
        # Draw the rectangle on the hdmap_np image
        cv2.rectangle(hdmap_np, top_left, bottom_right, (0, 0, 255), 5)  # BGR color format, red rectangle

    return hdmap_np


def add_nodes_and_edges(graph, object_centroids, distance_threshold):
    """
    This function enriches a graph by adding nodes representing objects, identified by their centroids, 
    and by establishing edges between these new nodes and existing ones based on proximity. 
    An edge is created only if the distance between nodes does not exceed a predefined threshold.

    Args:
    - graph (nx.Graph): The graph to which the nodes and edges will be added.
    - object_centroids (dict): A mapping of object identifiers to their centroid coordinates.
    - distance_threshold (float): The cutoff distance beyond which no edge is created between nodes.
    """
    import math

    # Add nodes with position attribute
    for obj, centroid in object_centroids.items():
        graph.add_node(obj, pos=centroid)
    
    # Calculate distances and add edges based on the distance threshold
    for new_node, new_node_pos in object_centroids.items():
        for existing_node, existing_node_attr in graph.nodes(data=True):
            if existing_node != new_node and str(existing_node).isdigit():  # Connect only to original nodes that are numbers
                existing_node_pos = existing_node_attr['pos']
                distance = math.sqrt((new_node_pos[0] - existing_node_pos[0])**2 + (new_node_pos[1] - existing_node_pos[1])**2)
                if distance <= distance_threshold:
                    graph.add_edge(new_node, existing_node)

    # For each new node without a numerical neighbor, find the closest numerical node and connect them
    for new_node in object_centroids.keys():
        if not any(str(neighbor).isdigit() for neighbor in graph.neighbors(new_node)):  # Check if it lacks a numerical neighbor
            closest_node = None
            closest_distance = float('inf')
            for existing_node, existing_node_attr in graph.nodes(data=True):
                if str(existing_node).isdigit():  # Only consider numerical nodes
                    existing_node_pos = existing_node_attr['pos']
                    distance = math.sqrt((object_centroids[new_node][0] - existing_node_pos[0])**2 + (object_centroids[new_node][1] - existing_node_pos[1])**2)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_node = existing_node
            if closest_node is not None:
                graph.add_edge(new_node, closest_node)


def create_text_centroid_dict(text_annotations):
    """
    Constructs a dictionary that associates detected text with the coordinates of its centroid.

    Parameters:
    - text_annotations: Annotations of text detected by the Google Vision API.

    Returns:
    - dict: A mapping where each detected text string is a key linked to a tuple representing its centroid coordinates.
    """
    
    def find_centroid(coordinates):
        """
        Finds the centroid of a bounding box given its four coordinates.

        Parameters:
        - coordinates (list of dicts): A list of four dictionaries, each representing a vertex of the bounding box.
          Each dictionary should have two keys: 'x' and 'y', corresponding to the vertex coordinates.

        Returns:
        - tuple: The (x, y) coordinates of the centroid.
        """
        x_coords = [vertex.x for vertex in coordinates]
        y_coords = [vertex.y for vertex in coordinates]

        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        return (centroid_x, centroid_y)

    text_centroid_dict = {}
    text_count = {}

    # Start from the second element to skip the summary entry
    for annotation in text_annotations[1:]:
        text = annotation.description
        # Check if the text has already been encountered
        if text in text_count:
            # Increment the count for the detected text
            text_count[text] += 1
            # Create a unique key for the text by appending its count
            unique_text_key = f"{text} {text_count[text]}"
        else:
            # Initialize the count for the detected text
            text_count[text] = 1
            # Use the original text as the key if it's the first instance
            unique_text_key = text

        centroid = find_centroid(annotation.bounding_poly.vertices)
        text_centroid_dict[unique_text_key] = centroid

    return text_centroid_dict


def generate_graph(centroids, threshold_percentage, indexed_color_image_resized_modified, start_corner='bottom_left'):
    """
    Generates a pruned graph from given centroids, applying a threshold percentage for pruning edges
    and reordering nodes from a specified corner in a diagonal ordering.

    Parameters:
    - centroids: A list of centroids where each centroid is a tuple or list of (y, x) coordinates.
    - threshold_percentage: The minimum percentage of a specific value required along the line between nodes for an edge to be retained.
    - indexed_color_image_resized_modified: The indexed color image after modification, used for determining the percentage of a specific value along the edge.
    - start_corner: The corner from which to start the node ordering ('top_left', 'bottom_left', 'top_right', 'bottom_right').

    Returns:
    - pruned_G: The pruned and reordered graph.
    """
    # Create a fully connected graph based on centroids
    G, pos = create_fully_connected_graph(centroids)
    max_distance = calculate_max_distance_for_edges(G, pos, weight_multiplier=1.5)
    
    # Prune graph edges based on the maximum distance and threshold percentage
    pruned_G = prune_graph_edges(G, max_distance, threshold_percentage, pos, indexed_color_image_resized_modified)
    
    # Determine sorting key based on the start_corner
    if start_corner == 'bottom_right':
        sort_key = lambda x: x[1][0] + x[1][1]
    elif start_corner == 'top_right':
        sort_key = lambda x: -x[1][0] + x[1][1]
    elif start_corner == 'bottom_left':
        sort_key = lambda x: x[1][0] - x[1][1]
    elif start_corner == 'top_left':
        sort_key = lambda x: -x[1][0] - x[1][1]
    else:
        raise ValueError("Invalid start_corner value. Choose from 'top_left', 'bottom_left', 'top_right', 'bottom_right'.")

    # Reordering nodes of pruned_G based on the specified corner
    sorted_nodes = sorted(pruned_G.nodes(data='pos'), key=sort_key)
    mapping = {node[0]: i for i, node in enumerate(sorted_nodes)}  # Create a mapping for node renaming
    
    # Relabel nodes based on the specified order
    pruned_G = nx.relabel_nodes(pruned_G, mapping)
    
    return pruned_G


def draw_graph_on_image(hdmap_image, pruned_G, output_path, nodes_to_keep):
    """
    Draws a graph on top of a hand-drawn map image and saves the result.

    Parameters:
    - hdmap_image: The hand-drawn map image (PIL Image or numpy array).
    - pruned_G: The pruned graph (networkx.Graph).
    - output_path: The file path to save the resulting image.
    - nodes_to_keep: List of numeric nodes to keep or ['all'] to keep all nodes.

    Returns:
    - None
    """
    # Make a copy of the pruned graph
    G_copy = pruned_G.copy()

    if nodes_to_keep != ['all']:
        # Identify all nodes to keep, including their non-numeric neighbors
        nodes_to_keep_set = set(nodes_to_keep)
        for node in nodes_to_keep:
            neighbors = list(G_copy.neighbors(node))
            for neighbor in neighbors:
                if not str(neighbor).isdigit():  # Check if the neighbor is non-numeric
                    nodes_to_keep_set.add(neighbor)
        
        # Remove nodes that are not in the nodes_to_keep_set
        nodes_to_remove = set(G_copy.nodes) - nodes_to_keep_set
        G_copy.remove_nodes_from(nodes_to_remove)

    # Plot the new graph on top of the image
    plt.imshow(hdmap_image)  # Display the hand-drawn map
    pos = nx.get_node_attributes(G_copy, 'pos')  # Get positions of all nodes
    numeric_nodes = {node for node in G_copy.nodes if str(node).isdigit()}
    non_numeric_nodes = set(G_copy.nodes) - numeric_nodes  # non numeric nodes are landmarks

    nx.draw_networkx_nodes(G_copy, pos, nodelist=numeric_nodes, node_color='orange', node_size=200)
    for node in non_numeric_nodes:  # make the non numeric nodes brown
        nx.draw_networkx_nodes(G_copy, pos, nodelist=[node], node_color='brown', node_shape='o', node_size=800)

    nx.draw_networkx_labels(G_copy, pos, font_size=5, font_color='black', labels={node: node for node in non_numeric_nodes})
    nx.draw_networkx_labels(G_copy, pos, font_size=7, font_color='black', labels={node: node for node in numeric_nodes})
    # nx.draw_networkx_edges(G_copy, pos, edge_color='black')

    plt.axis('tight')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":

    # define some variables
    n_colors = 4     # Let's assume the user specified 3 colors
    n_clusters = 39  # Number of clusters for free space (number of nodes)
    threshold_percentage = 98 # User-defined threshold for deciding if an edge should be kept based on pixel values

    # Load the image
    image_path = '/home/asblab/aaron/s3/process_hdmaps/hw_full.jpg'
    image = Image.open(image_path)

    # crop image until the first non white pixel is found, quantize it, and then downsize if necessary
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    image_np = np.array(cropped_image) # Convert the image into numpy array, image_np.shape = (height, width, 3)
    quantized_image_np = quantize_image(image_np, n_colors) # Quantize the image  
    target_width = quantized_image_np.shape[1]  # User specified width to downsize input image if necessary

    # Also calculate and print the number of unique colors after quantization
    unique_colors_after, counts_after = np.unique(quantized_image_np.reshape(-1, quantized_image_np.shape[2]), axis=0, return_counts=True)
    unique_colors_counts_after = {f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}": count for color, count in zip(unique_colors_after, counts_after)} # Dictionary to hold color codes (in hex) and counts
    # print("Unique colors and their counts after quantization: ", unique_colors_counts_after)

    # This block performs three main tasks:
    # 1. Initializes a 2D array 'indexed_color_image' with zeros, matching the dimensions of 'quantized_image_np', to store color indices for each pixel.
    # 2. Sorts the unique colors, ensuring white (ideally #fefefe) is positioned last in 'sorted_unique_colors_after' for consistent color indexing.
    # 3. Creates a dictionary 'color_to_index' mapping each unique color to an index, based on their sorted order in 'sorted_unique_colors_after'.
    indexed_color_image = np.zeros((quantized_image_np.shape[0], quantized_image_np.shape[1]), dtype=int)
    sorted_unique_colors_after = sorted(unique_colors_after, key=lambda color: (color != [254, 254, 254]).all())
    color_to_index = {tuple(color): idx for idx, color in enumerate(sorted_unique_colors_after)}
    
    # This section of code is responsible for two main tasks:
    # 1. Mapping each pixel in the quantized image to its corresponding color index and populating 'indexed_color_image' with these indices.
    # 2. Resizing the indexed color image to a specified target width while maintaining the aspect ratio.
    for i in range(quantized_image_np.shape[0]):
        for j in range(quantized_image_np.shape[1]):
            # Assign the index of the color of the current pixel to the indexed color image
            indexed_color_image[i, j] = color_to_index[tuple(quantized_image_np[i, j])]

    # Resize the indexed color image to the target width and save the result
    indexed_color_image_resized = create_indexed_color_image_at_resolution(quantized_image_np, target_width)
    # plt.imsave(f'/home/asblab/aaron/s3/process_hdmaps/resized_{target_width}.png', indexed_color_image_resized)

    # Find and plot centroids
    centroids = find_and_plot_centroids(indexed_color_image_resized, cell_value=2, n_clusters=n_clusters)
    indexed_color_image_resized_modified = indexed_color_image_resized.copy()
    indexed_color_image_resized_modified[indexed_color_image_resized == 2] = 3
    # print("centroids: ", centroids)
    # Save the indexed color image at the new resolution with the target_width in the filename
    # plt.imsave(f'/home/asblab/aaron/s3/process_hdmaps/resized_{target_width}_nopath.png', indexed_color_image_resized_modified)

    # Create a fully connected graph based on centroids
    pruned_G = generate_graph(centroids, threshold_percentage, indexed_color_image_resized_modified)

    # Creating a mask to identify pixels that are completely black ([0, 0, 0]) in the quantized image.
    # Then, replacing those black pixels with white ([254, 254, 254]) to ensure a consistent background.
    mask = (quantized_image_np == [0, 0, 0]).all(axis=-1)
    quantized_image_np[mask] = [254, 254, 254]
    # plt.imshow(quantized_image_np, cmap='viridis')
    # plt.axis('tight')
    # plt.axis('off')
    # plt.savefig(f'/home/asblab/aaron/s3/process_hdmaps/hdmap.png', bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # Draw the networkx graph on top of the matplotlib plot
    pos = nx.get_node_attributes(pruned_G, 'pos')
    # nx.draw(pruned_G, pos, with_labels=True, node_color='red', edge_color='black', node_size=300, font_size=10, font_color='white')
    # plt.savefig(f'/home/asblab/aaron/s3/process_hdmaps/hdmap_path_graph.png', bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close()

    # # LOAD THE IMAGE WITH PATH ONLY FOR VISUALIZATION PURPOSE -------------------------------------------------------------------
    # Load the image
    image_path = '/home/asblab/aaron/s3/process_hdmaps/hw_path.jpg'
    image = Image.open(image_path)

    # crop image until the first non white pixel is found, quantize it, and then downsize if necessary
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    image_np = np.array(cropped_image) # Convert the image into numpy array, image_np.shape = (height, width, 3)
    quantized_image_np = quantize_image(image_np, n_colors) # Quantize the image  
    target_width = quantized_image_np.shape[1]  # User specified width to downsize input image if necessary

    # Also calculate and print the number of unique colors after quantization
    unique_colors_after, counts_after = np.unique(quantized_image_np.reshape(-1, quantized_image_np.shape[2]), axis=0, return_counts=True)
    unique_colors_counts_after = {f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}": count for color, count in zip(unique_colors_after, counts_after)} # Dictionary to hold color codes (in hex) and counts
    # print("Unique colors and their counts after quantization: ", unique_colors_counts_after)

    # This block performs three main tasks:
    # 1. Initializes a 2D array 'indexed_color_image' with zeros, matching the dimensions of 'quantized_image_np', to store color indices for each pixel.
    # 2. Sorts the unique colors, ensuring white (ideally #fefefe) is positioned last in 'sorted_unique_colors_after' for consistent color indexing.
    # 3. Creates a dictionary 'color_to_index' mapping each unique color to an index, based on their sorted order in 'sorted_unique_colors_after'.
    indexed_color_image = np.zeros((quantized_image_np.shape[0], quantized_image_np.shape[1]), dtype=int)
    sorted_unique_colors_after = sorted(unique_colors_after, key=lambda color: (color != [254, 254, 254]).all())
    color_to_index = {tuple(color): idx for idx, color in enumerate(sorted_unique_colors_after)}
    
    # This section of code is responsible for two main tasks:
    # 1. Mapping each pixel in the quantized image to its corresponding color index and populating 'indexed_color_image' with these indices.
    # 2. Resizing the indexed color image to a specified target width while maintaining the aspect ratio.
    for i in range(quantized_image_np.shape[0]):
        for j in range(quantized_image_np.shape[1]):
            # Assign the index of the color of the current pixel to the indexed color image
            indexed_color_image[i, j] = color_to_index[tuple(quantized_image_np[i, j])]

    # Creating a mask to identify pixels that are completely black ([0, 0, 0]) in the quantized image.
    # Then, replacing those black pixels with white ([254, 254, 254]) to ensure a consistent background.
    mask = (quantized_image_np == [0, 0, 0]).all(axis=-1)
    quantized_image_np[mask] = [254, 254, 254]



    # # START GOOGLE VISION API-------------------------------------------------------------------
    image = vision.Image()
    hdmap_image = Image.fromarray(quantized_image_np)
    byte_io = io.BytesIO()
    hdmap_image.save(byte_io, format='PNG')
    # image_bytes = byte_io.getvalue()
    # image.content = image_bytes

    # # run text detection
    # vision_client = vision.ImageAnnotatorClient()
    # response = vision_client.text_detection(image=image)
    # text_annotations = response.text_annotations
    # # print("text annotations: ", text_annotations)
    # image_ = draw_bounding_boxes(quantized_image_np, text_annotations) # draw the bouding boxes
    # landmark_centroid_dict = create_text_centroid_dict(text_annotations) # create a dictionary of the landmarks (text from HDMap) and their centroids
    # print("landmark centroid dict: ", landmark_centroid_dict)

    # # Save the image with bounding boxes drawn
    # plt.imsave('/home/asblab/aaron/s3/process_hdmaps/hdmap_OCR_results.png', image_)
    # plt.close()

    # # # connect the landmarks to the pruned_G with edges
    # add_nodes_and_edges(pruned_G, landmark_centroid_dict, 300)

    # DRAW THE GRAPH WITH NODES AND EDGES -----------------------------------------------
    draw_graph_on_image(hdmap_image, pruned_G, '/home/asblab/aaron/s3/process_hdmaps/hdmap_path_landmarks.png', ['all'])

    # SAVE THE QUANTIZED IMAGE NP ARRAY AND GRAPH AS A PICKLE ---------------------------------
    # Save the quantized_image_np array to a pickle file
    pickle_file_path = '/home/asblab/aaron/s3/process_hdmaps/quantized_image_np.pkl'
    # print("Quantized image np size: ", quantized_image_np.shape)
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(quantized_image_np, file)
    print(f"quantized_image_np saved to {pickle_file_path}")

    # save pruned_G as a pickle
    pickle_file_path = '/home/asblab/aaron/s3/process_hdmaps/pruned_G.pkl'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(pruned_G, file)
    print(f"pruned_G saved to {pickle_file_path}")
