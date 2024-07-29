import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
import random
import heapq
from datetime import datetime as dt
from heapq import heappush, heappop
from functools import lru_cache
import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import os


# Assume boundaries and other setup code remains the same

@lru_cache(maxsize=None)
def do_lines_intersect_region(S, E, L1, L2, tolerance=1e-3):
    sx, sy, sz = S
    ex, ey, ez = E
    X1, Y1 = L1
    X2, Y2 = L2

    Xave = (X1 + X2) / 2
    Yave = (Y1 + Y2) / 2
    if np.isclose(X1, X2, atol=tolerance) and (sx - Xave) * (ex - Xave) < 0:
        minSEy, maxSEy = min(sy, ey) - tolerance, max(sy, ey) + tolerance
        minY, maxY = min(Y1, Y2) - tolerance, max(Y1, Y2) + tolerance
        return np.any(np.arange(minSEy, maxSEy, tolerance) >= minY) and np.any(
            np.arange(minSEy, maxSEy, tolerance) <= maxY)

    elif np.isclose(Y1, Y2, atol=tolerance) and (sy - Yave) * (ey - Yave) < 0:
        minSEx, maxSEx = min(sx, ex) - tolerance, max(sx, ex) + tolerance
        minX, maxX = min(X1, X2) - tolerance, max(X1, X2) + tolerance
        return np.any(np.arange(minSEx, maxSEx, tolerance) >= minX) and np.any(
            np.arange(minSEx, maxSEx, tolerance) <= maxX)

    return False

def is_passable(p1, p2, boundaries):
    node1 = tuple(p1)  # Convert numpy array to tuple
    node2 = tuple(p2)  # Convert numpy array to tuple
    return not any(do_lines_intersect_region(node1, node2, tuple(boundary[0]), tuple(boundary[1])) for boundary in boundaries)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def create_visibility_graph(df, boundaries):
    points = df[['X', 'Y', 'Z']].values
    graph = nx.Graph()
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points[i+1:], i+1):
            if is_passable(tuple(p1), tuple(p2), boundaries):  # Convert to tuple here
                dist = euclidean_distance(p1, p2)
                graph.add_edge(i, j, weight=dist)
    return graph
def parallel_shortest_path(args):
    G, source = args
    return source, nx.single_source_dijkstra_path_length(G, source)

def calculate_distance_matrix(df, boundaries):
    visibility_graph = create_visibility_graph(df, boundaries)
    nodes = list(visibility_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(visibility_graph, source) for source in nodes]
        results = list(executor.map(parallel_shortest_path, args))

    num_nodes = len(nodes)
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    for source, distances in results:
        source_idx = node_to_index[source]
        for target, dist in distances.items():
            target_idx = node_to_index[target]
            distance_matrix[source_idx, target_idx] = dist

    return distance_matrix

def christofides(distance_matrix):
    n = distance_matrix.shape[0]
    # Create a complete graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    # Compute minimum spanning tree
    mst = nx.minimum_spanning_tree(G)

    # Find odd-degree vertices
    odd_vertices = [v for v in mst.nodes() if mst.degree(v) % 2 == 1]

    # Compute minimum weight perfect matching
    matching = nx.min_weight_matching(G.subgraph(odd_vertices))

    # Combine MST and matching
    euler_graph = nx.MultiGraph(mst)
    euler_graph.add_edges_from(matching)

    # Find Eulerian circuit
    euler_circuit = list(nx.eulerian_circuit(euler_graph))

    # Make Hamiltonian circuit
    visited = set()
    hamiltonian_circuit = []
    for u, v in euler_circuit:
        if u not in visited:
            visited.add(u)
            hamiltonian_circuit.append(u)
    hamiltonian_circuit.append(hamiltonian_circuit[0])

    return hamiltonian_circuit


def find_optimal_path(df, shelves, start_shelf, boundaries):
    df_filtered = df[df['Regal/Fach/Boden'].isin(shelves + [start_shelf])]

    print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(df_filtered, boundaries)
    print("Distance matrix calculation complete.")

    print("Solving TSP...")
    optimal_path_indices = christofides(distance_matrix)
    print("TSP solved.")

    # Rotate the path to start with start_shelf
    start_index = df_filtered[df_filtered['Regal/Fach/Boden'] == start_shelf].index[0]
    start_pos = optimal_path_indices.index(start_index)
    optimal_path_indices = optimal_path_indices[start_pos:] + optimal_path_indices[1:start_pos]

    optimal_shelves = [df_filtered.iloc[i]['Regal/Fach/Boden'] for i in optimal_path_indices]
    optimal_distance = sum(distance_matrix[optimal_path_indices[i], optimal_path_indices[i + 1]]
                           for i in range(len(optimal_path_indices) - 1))

    return optimal_shelves, optimal_distance

def visualize_path_3d(df, boundaries, path):
    fig = go.Figure()

    # Add scatter plot for shelves (assuming Z is in df)
    fig.add_trace(go.Scatter3d(
        x=df['X'],
        y=df['Y'],
        z=df['Z'],
        mode='markers+text',
        marker=dict(size=5, color='blue'),
        # text=df['Regal/Fach/Boden'],
        text="",
        name='Shelves'
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

    plotly.offline.plot(fig)
# Main execution
if __name__ == '__main__':
    df = pd.read_excel(r'Stamm_Lagerorte.xlsx')

    shelves_to_visit = [
        # ['1/1/1', '1/2/1', '1/3/1'],
        ['3/1/1', '3/4/1', '4/2/1', '5/1/1', '21/2/1', '7/2/1'],
        # ['4/1/1', '4/1/7', '5/1/4'],
        # ['1/1/1', '1/4/1', '2/1/1', '3/4/1']
    ]

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

    start_shelf = '1/1/1'

    for shelves_list in shelves_to_visit:
        desired = shelves_list.copy()
        random.shuffle(shelves_list)

        start = dt.now()
        optimal_shelves, optimal_distance = find_optimal_path(df, shelves_list, start_shelf, boundaries)
        print("Time taken:", dt.now() - start)
        print("Shelves to Visit:", shelves_list, 'Desired Path:', desired, "Output Path:", optimal_shelves,
              'Output = Desired:', desired == optimal_shelves)


        visualize_path_3d(df, boundaries, [start_shelf] + optimal_shelves)