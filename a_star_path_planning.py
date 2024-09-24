## Path Planning with A*
# 23rd September 2024

''' A* Search Algorithm
1.  Initialize the open list

2.  Initialize the closed list put the starting node on the open list 
    (you can leave its f at zero)

3.  while the open list is not empty
    a) find the node with the least f on the open list, call it "q"
    b) pop q off the open list
    c) generate q's 8 successors and set their parents to q
    d) for each successor
        i) if successor is the goal, stop search
        ii) else, compute both g and h for successor
            successor.g = q.g + distance between successor and q

            successor.h = distance from goal to successor
            (This can be done using many ways, we will discuss three heuristics- Manhattan, Diagonal and Euclidean Heuristics)
          
            successor.f = successor.g + successor.h

        iii) if a node with the same position as successor is in the OPEN list which has a 
            lower f than successor, skip this successor

        iV) if a node with the same position as successor  is in the CLOSED list which has
            a lower f than successor, skip this successor otherwise, add  the node to the open list
     
     end (for loop)

    e) push q on the closed list
    
    end (while loop)

'''

'''
In this case, Euclidean would be the best since we can move in any direction

'''


## IMPORT THE REQUIRED LIBRARIES
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import copy

printingFlag = False

# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

# Define the size of the grid
ROW = 33
COL = 33

# Check if a cell is valid (within the grid)
def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

# Check if a cell is unblocked
def is_unblocked(grid, row, col):
    return grid[row][col] == 1

# Check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# Trace the path from source to destination
def trace_path(cell_details, dest):
    if printingFlag:
        print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    path_coord = convert_grid_to_coord(path).tolist()
    

    # Print the path
    if printingFlag:
        for i in path:
            print("->", i, end=" ")
            
        print()


    # # Print the path
    # for i in path_coord:
    #     print("->", i, end=" ")
        
    # print()

    return path

# Implement the A* search algorithm
def a_star_search(grid, src, dest):
    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are unblocked
    if (not is_unblocked(grid, src[0], src[1])) or (not is_unblocked(grid, dest[0], dest[1])):
        print("Source or the destination is blocked")
        # print(f"Source: {src}, Dest: {dest}")
        # print("src: ", is_unblocked(grid, src[0], src[1]))
        # print("dest: ", is_unblocked(grid, dest[0], dest[1]))
        # print(f"Source blocked: {grid[src[0]][src[1]]}, Dest blocked: {grid[dest[0]][dest[1]]}")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    if printingFlag:
                        print("The destination cell is found")
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)

                    path_coord = convert_grid_to_coord(path)
                    
                    x_points = []
                    y_points = []
                    for i in range(len(path_coord)):
                        x_points.append(path_coord[i][0])
                        y_points.append(path_coord[i][1])
                    
                    plt.plot(x_points, y_points, marker = 'o') 
                    # space = [round(i*0.1, 2) for i in range(-16, 17, 1)]
                    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])  
                    plt.xticks(space); plt.yticks(space)
                    for i, (x, y) in enumerate(zip(x_points, y_points), 1):
                        plt.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
                    plt.grid()
                    
                    print(f'Path found: {path_coord}')
                    found_dest = True
                    return path_coord
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")



def modify_obstacles(aruco_true_pos, search_fruit_ind, fruits_true_pos):

    print("True Position of fruits in modiy, at the start: ", fruits_true_pos)

    grid = np.ones([ROW, COL], dtype = int)

    obstacles_list = []
    fruits_copy = copy.deepcopy(fruits_true_pos)
    fruits_copy.pop(search_fruit_ind)
    obstacles_list = aruco_true_pos + fruits_copy

    obstacle_grid  = []
    for i in range (len(obstacles_list)):
        obstacle_coord = convert_coord_to_grid(obstacles_list[i])
        obstacle_grid.append(obstacle_coord)


    for i in range(len(obstacle_grid)):
        grid[obstacle_grid[i][1], obstacle_grid[i][0]] = 0

    # for i in range(ROW):
    #     for j in range(COL):
    #         # print(grid[i][j], end=' ')
    #         print(f"Y: {i} X: {j} Value: {grid[j][i]}")
        # print()

    return grid


def convert_coord_to_grid(coord_dest):
    space = [round(i*0.1, 2) for i in range(-16, 17, 1)]
    grid_dest = []
    for i in range(len(coord_dest)):
        for j in range(len(space)):
            if coord_dest[i] == space[j]:
                grid_dest.append(j)

    
    return grid_dest

def convert_grid_to_coord(path):
    space = [round(i*0.1, 2) for i in range(-16, 17, 1)]
    path_coord = np.zeros([len(path), 2])

    for i in range(len(space)):
        for j in range(len(path)): # y
            if path[j][1] == i:
                path_coord[j][1] = space[i]
            for k in range(len(path)): # x
                if path[k][0] == i:
                    path_coord[k][0] = space[i]
                
    return path_coord


def main():
    fruits_list=  ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
    fruits_true_pos=  [[-0.4,  0  ],
                       [ 1.4, -0.8],
                       [ 0.8, -0.4],
                       [-1.2, -1.2],
                       [ 0.8,  0.8]]

    aruco_true_pos=  [[   0,  0.2],
                      [ 1.2,  0.4],
                      [-1.2,  1  ],
                      [-1  , -0.4],
                      [-1  , -1  ],
                      [-1  ,  0.2],
                      [ 0.4, -0.8],
                      [ 0.4,  0  ],
                      [-0.4, -0.6],
                      [ 1.6, -0.4]]
    
    search_list =   ['redapple', 'greenapple', 'orange']
    search_index = []


    for i in range(len(search_list)):          ## The shopping list only, so 3
        for j in range(len(fruits_list)):      ## The full list at 5
            if search_list[i] == fruits_list[j]:
                search_index.append(j)
    
    search_true_pos = []

    for i in range(len(search_index)):
        search_true_pos.append(fruits_true_pos[search_index[i]])

    fruits_copy = copy.deepcopy(fruits_true_pos)
    src = [0,0]
    for i in range(len(search_list)):
        # Run the A* search algorithm
        
        grid_src = convert_coord_to_grid(src)
        dest = search_true_pos[i]
        
        for j in range (len(dest)):
            if dest[j] < 0:
                dest[j] += 0.1
                
            else:
                dest[j] -= 0.1
            dest[j] = round(dest[j], 2)
        # print(dest)
        grid_dest = convert_coord_to_grid(dest)

        grid = modify_obstacles(aruco_true_pos, search_index[i], fruits_true_pos)
        
        waypoints = a_star_search(grid, grid_src, grid_dest)

        src = dest
    
    x_aruco = []
    y_aruco = []
    x_fruits = []
    y_fruits = []

    for i in range(len(aruco_true_pos)):
        x_aruco.append(aruco_true_pos[i][0])
        y_aruco.append(aruco_true_pos[i][1])
    
    plt.plot(x_aruco, y_aruco, 'ok') 
    for i, (x, y) in enumerate(zip(x_aruco, y_aruco), 1):
        plt.annotate(f'aruco_{i}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
    
    # fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]
    fruit_colour = ["red", "cyan", "orange", "yellow", "green"]

    for i in range(len(fruits_copy)):
        x_fruits.append(fruits_copy[i][0])
        y_fruits.append(fruits_copy[i][1])
    plt.scatter(x_fruits, y_fruits, c=fruit_colour, s=100)
    for i, (x, y) in enumerate(zip(x_fruits, y_fruits), 1):
        plt.annotate(f'{i+10}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
    

    plt.show()
    

if __name__ == "__main__":
    main()