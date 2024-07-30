import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go
import random
import heapq
import itertools

df = pd.read_excel(r'Stamm_Lagerorte.xlsx')  # Import Dataframe


# Define a function to check if two line segments intersect with a given tolerance
def do_lines_intersect(p1, p2, q1, q2, tolerance=1e-5):
    # Helper function to determine if three points are in a counterclockwise order
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Helper function to check if a point (px, py) lies on the segment (ax, ay) - (bx, by)
    # considering a tolerance for floating-point precision issues
    def point_on_segment(px, py, ax, ay, bx, by):
        return min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance and \
            min(ay, by) - tolerance <= py <= max(ay, by) + tolerance

    # Check if the line segments (p1, p2) and (q1, q2) intersect by using the CCW test
    if ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2):
        return True

    # Additionally, check if any endpoint of one segment lies on the other segment
    # This accounts for the case when segments are collinear or overlapping
    if point_on_segment(p1[0], p1[1], q1[0], q1[1], q2[0], q2[1]) or \
            point_on_segment(p2[0], p2[1], q1[0], q1[1], q2[0], q2[1]) or \
            point_on_segment(q1[0], q1[1], p1[0], p1[1], p2[0], p2[1]) or \
            point_on_segment(q2[0], q2[1], p1[0], p1[1], p2[0], p2[1]):
        return True

    # If neither condition is met, the segments do not intersect
    return False


def is_passable(node1, node2, boundaries):
    for boundary in boundaries:
        if do_lines_intersect(node1, node2, boundary[0], boundary[1]):
            return False
    return True


def heuristic(a, b):
    # Manhattan distance heuristic function
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start, goal, boundaries):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if np.all(np.isclose(current, goal, atol=1e-5)):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01)]]
        for neighbor in neighbors:
            if not is_passable(current, neighbor, boundaries):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


# Function to calculate the Manhattan distance matrix with boundary consideration using A* algorithm
def calculate_manhattan_distance_matrix_with_boundaries(df, boundaries):
    coords = df[['X', 'Y', 'Z']].values
    aisles = df['Aisle'].values
    num_shelves = len(coords)
    distance_matrix = np.zeros((num_shelves, num_shelves))

    for i in range(num_shelves):
        for j in range(i, num_shelves):
            if i == j:
                distance_matrix[i, j] = np.inf  # Avoid zero distance to itself
            else:  # i < j
                path = a_star((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1]), boundaries)
                if path:
                    manhattan_distance = len(path) - 1
                    aisle_penalty = 1000 * np.abs(aisles[i] - aisles[j])  # Adjust the penalty as needed
                    distance_matrix[i, j] = manhattan_distance + aisle_penalty
                    # print(f"({coords[i][0]},{coords[i][1]}) => ({coords[j][0]},{coords[j][1]}) : {distance_matrix[i, j]}")
                else:
                    distance_matrix[i, j] = np.inf  # No valid path found
                distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


# Nearest Neighbor Algorithm with a fixed starting point
def nearest_neighbor_tsp(distance_matrix, start_index):
    num_shelves = distance_matrix.shape[0]
    visited = [False] * num_shelves
    path = [start_index]
    visited[start_index] = True
    current = start_index

    while len(path) < num_shelves:
        nearest = np.argmin([distance_matrix[current, j] if not visited[j] else np.inf for j in range(num_shelves)])
        path.append(nearest)
        visited[nearest] = True
        current = nearest

    path.append(start_index)  # Return to the starting point
    total_distance = sum(distance_matrix[path[i], path[i + 1]] for i in range(num_shelves))
    return path, total_distance


def calculate_route_distance(route, distance_matrix):
    """Calculate the total distance of the given route based on the distance matrix."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


# Function to find the optimal path for a given list of shelves with a fixed starting point
def find_optimal_path(df, shelves, start_shelf, boundaries):
    # Add starting shelf if not present
    start = True
    if start_shelf not in shelves:
        start = False
        shelves.append(start_shelf)

    # Filter the DataFrame based on the list of shelves
    df_filtered = df[df['Regal/Fach/Boden'].isin(shelves)]

    # Calculate the Manhattan distance matrix with boundary consideration
    distance_matrix = calculate_manhattan_distance_matrix_with_boundaries(df_filtered, boundaries)
    # print(distance_matrix)

    print(len(df_filtered))
    df_filtered_reset = df_filtered.reset_index(drop=True)
    # Find the index of the starting shelf
    start_index = df_filtered_reset[df_filtered_reset['Regal/Fach/Boden'] == start_shelf].index[0]

    print("Start Index, ", start_index)

    # Solve the TSP using Nearest Neighbor Algorithm with a fixed starting point
    optimal_path_indices, optimal_distance = nearest_neighbor_tsp(distance_matrix, start_index)

    # Map the optimal path back to the shelf coordinates
    optimal_shelves = [df_filtered.iloc[i]['Regal/Fach/Boden'] for i in optimal_path_indices]

    if not start:
        optimal_shelves = optimal_shelves[1:-1]  # Remove the starting shelf from the path
    else:
        optimal_shelves = optimal_shelves[:-1]

    return optimal_shelves, optimal_distance


def distance(a, b):
    return np.linalg.norm(np.array(b) - np.array(a))


# def assign_scores(df, start_shelf, boundaries):
#
#     for i,r in df.iterrows():
#         start = (df[df['Regal/Fach/Boden'] == (start_shelf)][['X', 'Y', 'Z']].values)[0]
#         current = (df[df['Regal/Fach/Boden'] == (r['Regal/Fach/Boden'])][['X', 'Y', 'Z']].values)[0]
#
#         Xmin = min(min(x1, x2) for (x1, y1), (x2, y2) in boundaries)
#         Xmax = max(max(x1, x2) for (x1, y1), (x2, y2) in boundaries)
#         Xave = (Xmax + Xmin) / 2
#         scoreAisle = abs(Xave - current[0]) * 10
#         scoreDist = distance(start, current) * 100
#         scoreGround = current[2] * 10
#         score = scoreDist * 0.33 + scoreAisle * 0.33 + scoreGround * 0.33
#
#
#
#
#         df.at[i, 'Score'] = score
#
#     return df

def assign_scores(df, start_shelf, boundaries):
    # Step 1: Calculate all distances from the aisle and store them
    aisle_distances = []
    start_distances = []

    # Find the middle of the aisle
    Xmin = min(min(x1, x2) for (x1, y1), (x2, y2) in boundaries)
    Xmax = max(max(x1, x2) for (x1, y1), (x2, y2) in boundaries)
    Xave = (Xmax + Xmin) / 2

    # Find the starting point
    start = df[df['Regal/Fach/Boden'] == start_shelf][['X', 'Y', 'Z']].values[0]

    for i, r in df.iterrows():
        current = df.loc[i, ['X', 'Y', 'Z']].values
        distance_from_aisle = abs(Xave - current[0])
        distance_from_start = distance(start, current)

        aisle_distances.append(distance_from_aisle)
        start_distances.append(distance_from_start)

    # Normalize the aisle distances to a 0-100 scale
    max_aisle_distance = max(aisle_distances)
    min_aisle_distance = min(aisle_distances)
    normalized_aisle_scores = [(1 - (d - min_aisle_distance) / (max_aisle_distance - min_aisle_distance)) * 100 for d in aisle_distances]

    # Normalize the start distances to a 0-100 scale
    max_start_distance = max(start_distances)
    min_start_distance = min(start_distances)
    normalized_start_scores = [(1 - (d - min_start_distance) / (max_start_distance - min_start_distance)) * 100 for d in start_distances]

    # Step 2: Calculate the scores
    for i, r in df.iterrows():
        current = df.loc[i, ['X', 'Y', 'Z']].values

        # Ground score
        if current[2] < 0.3:
            scoreGround = 0
        elif (current[2] > 0.3) and (current[2] < 0.6):
            scoreGround =25

        elif (current[2] > 0.6) and (current[2] <0.9):
            scoreGround = 50
        elif (current[2] > 0.9) and (current[2] < 1.1):
            scoreGround = 75
        elif (current[2] > 1.1) and (current[2]< 1.5):
            scoreGround = 100
        elif (current[2] > 1.5) and (current[2]< 1.7):
            scoreGround = 50
        elif (current[2] > 1.7) and (current[2] < 1.9):
            scoreGround = 25
        elif current[2] > 1.9:
            scoreGround = 0

        # Get the normalized aisle and start scores
        scoreAisle = normalized_aisle_scores[i]
        scoreDist = normalized_start_scores[i]

        # Calculate the final score
        score = scoreDist * 0.33 + scoreAisle * 0.33 + scoreGround * 0.33


        # print(r, score, scoreGround, scoreAisle, scoreDist)

        # Assign the score to the dataframe
        df.at[i, 'Score'] = score

    return df

# Function to visualize the shelves, boundaries, and path
def visualize_path_3d(df, boundaries, path, start_shelf):
    fig = go.Figure()

    # Normalize scores for color mapping
    norm = Normalize(vmin=df['Score'].min(), vmax=df['Score'].max())
    cmap = plt.get_cmap('RdYlGn')  # Red to Green colormap

    # Add scatter plot for shelves (assuming Z is in df)
    for index, shelf in df.iterrows():
        color = cmap(norm(shelf['Score']))
        color_rgb = f'rgb({color[0] * 255}, {color[1] * 255}, {color[2] * 255})'
        fig.add_trace(go.Scatter3d(
            x=[shelf['X']],
            y=[shelf['Y']],
            z=[shelf['Z']],
            mode="markers+text",
            marker=dict(
                size=5,
                color=color_rgb
            ),
            text=shelf['Regal/Fach/Boden'],
            name="Shelves"
        ))

    # Assume a fixed Z value for all boundary points
    z_value = 0  # You can set this to any constant value appropriate for your visualization

    # Add lines for boundaries
    for boundary in boundaries:
        fig.add_trace(go.Scatter3d(
            x=[boundary[0][0], boundary[1][0]],
            y=[boundary[0][1], boundary[1][1]],
            z=[z_value, z_value],  # Fixed Z value for boundaries
            mode='lines',
            line=dict(color='red', width=2),
            name='Boundary'
        ))

    # Add path lines
    for i in range(len(path) - 1):
        start_shelf = df[df['Regal/Fach/Boden'] == path[i]]
        end_shelf = df[df['Regal/Fach/Boden'] == path[i + 1]]
        fig.add_trace(go.Scatter3d(
            x=[start_shelf['X'].values[0], end_shelf['X'].values[0]],
            y=[start_shelf['Y'].values[0], end_shelf['Y'].values[0]],
            z=[start_shelf['Z'].values[0], end_shelf['Z'].values[0]],
            mode='lines',
            line=dict(color='green', width=2),
            name=f'Path {i}'
        ))

    # Set plot layout
    fig.update_layout(
        title='3D Visualization of Warehouse Paths',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate'
        ),
        autosize=True
    )

    fig.show()


# Function to visualize the shelves, boundaries, and path
def visualize_path(df, boundaries, path):
    plt.figure(figsize=(10, 8))

    # Plot shelves
    plt.scatter(df['X'], df['Y'], c='blue', label='Shelves')
    for i, txt in enumerate(df['Regal/Fach/Boden']):
        plt.annotate(txt, (df['X'].iloc[i], df['Y'].iloc[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Plot boundaries
    for boundary in boundaries:
        plt.plot([boundary[0][0], boundary[1][0]], [boundary[0][1], boundary[1][1]], 'r-', label='Boundary')

    # Plot path
    for i in range(len(path) - 1):
        start_shelf = df[df['Regal/Fach/Boden'] == path[i]]
        end_shelf = df[df['Regal/Fach/Boden'] == path[i + 1]]
        plt.plot([start_shelf['X'].values[0], end_shelf['X'].values[0]],
                 [start_shelf['Y'].values[0], end_shelf['Y'].values[0]], 'g-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right')
    plt.title('Optimal Path Visualization')
    plt.show()


# Define boundaries as a list of line segments (start and end points)
boundaries = [
    # Walls
    ((0, 0), (0, 12)),
    ((0, 12), (7, 12)),
    ((0, 0), (7, 0)),
    ((7, 0), (7, 12)),
    # Shelves
    ((0, 3.7), (3, 3.7)),
    ((0, 5), (3, 5)),
    ((0, 6.6), (3, 6.6)),
    ((0, 8), (3, 8)),
    ((0, 9.5), (3, 9.5)),
    ((5, 9.65), (7, 9.65)),
    ((5, 7), (7, 7)),
    ((5, 5.5), (7, 5.5)),
]

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    # Example usage
    shelves_to_visit = [
        # ['1/1/1', '1/2/1', '1/3/1'],
        # ['3/1/1', '3/4/1', '4/2/1', '5/1/1', '21/2/1', '7/2/1','14/2/1'],
        ['4/1/1', '4/1/7', '5/1/4'],
        # ['1/1/1', '1/4/1', '2/1/1', '3/4/1']
    ]

    start_shelf = 'Startpoint'

    df.loc[df['Regal/Fach/Boden'] == 'Startpoint', 'X'] = 4
    df.loc[df['Regal/Fach/Boden'] == 'Startpoint', 'Y'] = 2


    #in df row where regal is Staretpoint replace x value
    df = assign_scores(df,start_shelf,boundaries)
    # print(df.tail())
    #
    # breakpoint()



    for shelves_list in shelves_to_visit:
        # Randomly shuffle list
        desired = shelves_list.copy()
        random.shuffle(shelves_list)

        optimal_shelves, optimal_distance = find_optimal_path(df, shelves_list.copy(), start_shelf, boundaries)
        print("Shelves to Visit:", shelves_list, 'Desired Path:', desired, "Output Path:", optimal_shelves,
              'Output = Desired:', desired == optimal_shelves)

        # Visualize the result
        visualize_path_3d(df, boundaries, [start_shelf] + optimal_shelves, start_shelf)