from helper import *
from vision_utils import *
from lmm_localize import *


def get_position(node, graph):
    """
    Retrieve the position of a node within a graph.

    Args:
    node: The node identifier within the graph.
    graph: The graph data structure containing nodes and their attributes.

    Returns:
    tuple: The position of the node as a tuple (x, y).
    """
    return graph.nodes[node]['pos']


def find_direction(prev_pos, curr_pos):
    """
    Determine the cardinal direction of movement from a previous position to a current position.

    Args:
    prev_pos (tuple): The previous position as a tuple (x, y).
    curr_pos (tuple): The current position as a tuple (x, y).

    Returns:
    str or None: Returns 'N', 'S', 'E', 'W' for north, south, east, and west respectively,
                 or None if there is no movement in either direction.
    """
    # Calculate the differences in the x and y coordinates
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    
    # Determine if movement is more significant horizontally or vertically
    if abs(dx) > abs(dy):
        # Movement is primarily horizontal
        if dx > 0:
            return 'E'  # East if x increased
        else:
            return 'W'  # West if x decreased
    else:
        # Movement is primarily vertical
        if dy > 0:
            return 'S'  # South if y increased (down in many graphical representations)
        elif dy < 0:
            return 'N'  # North if y decreased
        else:
            return None  # No significant movement in either direction


def direction_to_action(directions):
    """
    Converts a sequence of cardinal directions into a list of actions such as turning or going straight.

    Args:
    directions (list of str): A list of cardinal directions ('N', 'S', 'E', 'W') representing the path taken.

    Returns:
    list of str: A list of actions ('TURN_RIGHT', 'TURN_LEFT', 'GO_STRAIGHT', 'unknown') corresponding to the transitions between directions.
    """
    
    # Mapping of direction transitions to turn actions
    turns = {
        ('N', 'E'): "TURN_RIGHT",
        ('E', 'S'): "TURN_RIGHT",
        ('S', 'W'): "TURN_RIGHT",
        ('W', 'N'): "TURN_RIGHT",
        ('N', 'W'): "TURN_LEFT",
        ('W', 'S'): "TURN_LEFT",
        ('S', 'E'): "TURN_LEFT",
        ('E', 'N'): "TURN_LEFT",
    }
    
    # Initialize the actions list with 'GO_STRAIGHT' assuming the first action is to move straight
    actions = ["GO_STRAIGHT"]
    
    # Iterate through the list of directions to determine the necessary actions
    for i in range(1, len(directions)):
        prev_dir, curr_dir = directions[i-1], directions[i]
        
        # Check if the direction has not changed
        if prev_dir == curr_dir:
            action = "GO_STRAIGHT"
        else:
            # Determine the turn action based on the transition from previous to current direction
            action = turns.get((prev_dir, curr_dir), "unknown")  # Default to "unknown" if transition is not defined
        
        # Append the determined action to the actions list
        actions.append(action)
    
    # Return the complete list of actions
    return actions


def find_shift_indices(actions):
    """
    Identifies the indices in the actions list where a change in action occurs, indicating a shift.

    This function scans through the list of actions and determines where the action changes from one type to another.
    These indices represent points where a shift in action is required, such as changing from going straight to turning.

    Args:
    actions (list of str): A list of actions (e.g., 'TURN_RIGHT', 'GO_STRAIGHT').

    Returns:
    list of int: A list of indices indicating where shifts in actions occur.
    """
    # Initialize an empty list to store the indices of action shifts
    shift_indices = []
    
    # Iterate through the list of actions starting from the second element (index 1)
    for i in range(1, len(actions)):
        # Check if the current action is different from the previous one
        # and also ensure that the current action is not already the start of a new shift
        if actions[i] != actions[i-1] and (not shift_indices or i - 1 != shift_indices[-1]):
            # If the conditions are met, append the current index to the shift_indices list
            shift_indices.append(i)
    
    # Return the list of indices where shifts occur
    return shift_indices


def print_non_digit_neighbors_at_shifts(actions, sorted_nodes, hdmap_graph):
    """
    Prints nodes where action shifts occur along with their non-digit neighbors.

    This function identifies the nodes at which the action changes (shift nodes) and prints these nodes,
    the corresponding actions, and their neighbors that are not represented by digits.

    Args:
    actions (list of str): A list of actions taken at each node.
    sorted_nodes (list): A list of nodes sorted in a specific order.
    hdmap_graph (Graph): A graph structure representing the nodes and their connections.

    Returns:
    tuple: A tuple containing the list of shift nodes and their indices.
    """
    # Find indices in the actions list where the action changes
    shift_indices = find_shift_indices(actions)
    # Retrieve the nodes at these indices
    shift_nodes = [sorted_nodes[i] for i in shift_indices]
    
    # Print the shift nodes and their indices for debugging or information purposes
    # print("shift nodes: ", shift_nodes)
    # print("shift indices: ", shift_indices)

    # # Iterate through each shift node to find and print non-digit neighbors
    # for node in shift_nodes:
    #     # Get all neighbors of the node
    #     neighbors = list(hdmap_graph.neighbors(node))
    #     # Filter out neighbors that are not represented by digits
    #     non_digit_neighbors = [n for n in neighbors if not str(n).isdigit()]
    #     # Find the index of the current node in the list of shift nodes
    #     action_index = shift_nodes.index(node)
    #     # Get the action at this index, default to "continue" if index is out of bounds
    #     action = actions[action_index] if action_index < len(actions) else "continue"
    #     # Print the node, its action, and its non-digit neighbors
    #     print(f"Node: {node}, Action: {action}, Non-Digit Neighbors: {non_digit_neighbors}")

    # Return the shift nodes and their indices
    return shift_nodes, shift_indices


def generate_node_details(nodes, shift_nodes, shift_indices, actions, node_structure_dict):
    """
    Generates detailed information for each node, including actions and landmarks.

    This function processes a list of nodes and determines the action to be taken at each node
    and identifies non-digit landmarks (neighbors) associated with each node.

    Args:
    nodes (list): A list of all nodes to process.
    shift_nodes (list): A list of nodes where an action shift occurs.
    shift_indices (list of int): A list of indices indicating where shifts in actions occur.
    actions (list of str): A list of actions corresponding to each node in shift_nodes.
    node_structure_dict (dict): A dictionary containing landmark descriptions for each node.

    Returns:
    dict: A dictionary where each key is a node and each value is another dictionary with keys 'action' and 'landmarks'.
    """
    shift_actions = {node: actions[index] for node, index in zip(shift_nodes, shift_indices)}
    node_details = {}

    for node in nodes:
        if node in shift_nodes:
            action = shift_actions[node]
        else:
            action = 'GO_STRAIGHT'

        # Include landmark description from node_structure_dict
        landmark_description = node_structure_dict.get(node, {}).get('Landmarks', 'when possible')

        node_details[node] = {"action": action, "landmarks": [landmark_description]}

    return node_details


def get_navigation_segments(sorted_nodes, shift_nodes, node_details):
    """
    Creates a sequence of segment descriptions utilizing node details and shift points.

    This function processes nodes in sorted order to generate a sequence of segments. Each segment
    details the actions and landmarks from one shift node to the next or until the end of the route.
    Args:
    sorted_nodes (list): A list of nodes sorted in the order they are encountered.
    shift_nodes (list): A list of nodes where there is a change in action.
    node_details (dict): A dictionary containing details about each node, including actions and landmarks.

    Returns:
    list: A list of strings, each describing a segment of the route.
    """
    segments = []  # Initialize the list to hold segment descriptions.
    start_index = 0  # Initialize the start index for the first segment.

    # Iterate through each node in sorted_nodes using its index and value.
    for i, node in enumerate(sorted_nodes):
        # Check if the current node is a shift node or the last node in the list.
        if node in shift_nodes or i == len(sorted_nodes) - 1:
            # Slice the sorted_nodes to get the current segment including the current node.
            segment_nodes = sorted_nodes[start_index:i+1]
            segment_landmarks = []  # Initialize a list to collect landmarks for this segment.

            # Collect landmarks from each node in the segment except the last one.
            for segment_node in segment_nodes[:-1]:  # Exclude the last node to avoid repetition.
                landmarks = node_details[segment_node]['landmarks']
                # Only add landmarks if they are not the default placeholder "when possible".
                if landmarks != "when possible":
                    segment_landmarks.extend(landmarks)

            # Remove duplicate landmarks while preserving the order using a set.
            seen = set()
            segment_landmarks = [x for x in segment_landmarks if not (x in seen or seen.add(x))]

            # Construct the segment action description based on the collected landmarks.
            if segment_landmarks and segment_landmarks != ["when possible"]:
                segment_action = f"GO_STRAIGHT pass {segment_landmarks}"
            else:
                segment_action = f"GO_STRAIGHT until {node_details[node]['landmarks']}"
            segments.append(segment_action)  # Add the segment description to the list.

            # Handle specific actions at shift nodes or the last node.
            if node in shift_nodes:
                # Add the action at the shift node using its specific action and landmarks.
                shift_action = f"{node_details[node]['action']} at {node_details[node]['landmarks']}"
                segments.append(shift_action)
            elif i == len(sorted_nodes) - 1:
                # Specifically handle the last node with a STOP action.
                stop_action = f"STOP at {node_details[node]['landmarks']}"
                segments.append(stop_action)

            # Update the start_index for the next segment to start from the next node.
            start_index = i + 1

    return segments  # Return the list of segment descriptions.


def get_node_direction_map(hdmap_graph):
    """
    Generates a mapping of nodes to their respective directions based on the hdmap_graph.

    Args:
    hdmap_graph (Graph): A graph structure representing the nodes and their connections.

    Returns:
    dict: A dictionary where keys are node identifiers and values are directions ('N', 'S', 'E', 'W').
    """

    # Filter nodes to include only those that are digits
    digit_nodes = [node for node in hdmap_graph.nodes() if str(node).isdigit()]
    sorted_nodes = sorted(digit_nodes, key=lambda x: int(x))  # Sort the filtered nodes by their labels
    sorted_nodes.reverse()  # reverse the sorted_nodes to go from starting point (9) to ending point (0)

    # Get the direction from the previous node to the current node
    directions = []
    for i in range(1, len(sorted_nodes)):
        prev_node, curr_node = sorted_nodes[i-1], sorted_nodes[i]
        prev_pos = get_position(prev_node, hdmap_graph)
        curr_pos = get_position(curr_node, hdmap_graph)
        direction = find_direction(prev_pos, curr_pos)
        if direction:
            directions.append(direction)

    directions.insert(0, "N") # insert the robot's starting direction

    # get the numeric nodes and sort them
    numeric_nodes = sorted([node for node in hdmap_graph.nodes if isinstance(node, int)], reverse=True)
    node_direction_map = {node: direction for node, direction in zip(numeric_nodes, directions)} # Create a dictionary matching nodes to directions
    
    return node_direction_map


def generate_global_navigation_plan(hdmap_graph, node_structure_dict):
    """
    Generates a global navigation plan based on the hdmap_graph provided.

    Args:
    hdmap_graph (Graph): A graph structure representing the nodes and their connections.
    node_structure_dict (dict): A dictionary containing landmark descriptions for each node.

    Returns:
    list: A list of navigation instructions.
    """
    # Filter nodes to include only those that are digits
    digit_nodes = [node for node in hdmap_graph.nodes() if str(node).isdigit()]
    sorted_nodes = sorted(digit_nodes, key=lambda x: int(x))  # Sort the filtered nodes by their labels
    sorted_nodes.reverse()  # reverse the sorted_nodes to go from starting point (9) to ending point (0)

    # Get the direction from the previous node to the current node
    directions = []
    for i in range(1, len(sorted_nodes)):
        prev_node, curr_node = sorted_nodes[i-1], sorted_nodes[i]
        prev_pos = get_position(prev_node, hdmap_graph)
        curr_pos = get_position(curr_node, hdmap_graph)
        direction = find_direction(prev_pos, curr_pos)
        if direction:
            directions.append(direction)

    actions = direction_to_action(directions)
    shift_nodes, shift_indices = print_non_digit_neighbors_at_shifts(actions, sorted_nodes, hdmap_graph)
    node_details = generate_node_details(sorted_nodes, shift_nodes, shift_indices, actions, node_structure_dict)
    global_plan = get_navigation_segments(sorted_nodes, shift_nodes, node_details)

    return global_plan, shift_nodes


if __name__=="__main__":
    # Load the HD map data, including the numpy array representation, the graph structure, and the current directory
    hdmap_np, hdmap_graph, current_dir = load_hdmap_data()
    
    # Print the neighbors of each node in the graph for debugging purposes
    for node in hdmap_graph.nodes:
        neighbors = list(hdmap_graph.neighbors(node))
        print(f"Node: {node}, Neighbors: {neighbors}")

    # Generate a mapping of nodes to their respective directions
    node_direction_map = get_node_direction_map(hdmap_graph)
    
    # Generate the node structure and a dictionary containing landmark descriptions for each node
    node_structure, node_structure_dict = generate_node_structure(hdmap_graph, node_direction_map)
    
    # Generate the global navigation plan based on the HD map graph and node structure dictionary
    global_plan, _ = generate_global_navigation_plan(hdmap_graph, node_structure_dict)

    # Print the global navigation instructions
    print()
    print("Global Navigation Instructions:")
    for local_plan in global_plan:
        print(local_plan)

