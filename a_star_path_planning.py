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
Heuristic: Euclidean, since we can move in any direction 
'''



## IMPORT THE REQUIRED LIBRARIES
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import copy
import json
import ast

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

# Defining Global Variables
global printingFlag, printingTestFlag, plotFlag, modificationFlag
global how_far_from_fruits, impactRadiusSize, spacing, divisor, thresholdDistance
global segementedFile
printingFlag = False
printingTestFlag = False
plotFlag = True
modificationFlag = False

how_far_from_fruits = 0.3
impactRadiusSize = 0.1
spacing = 10
divisor = 0.01
thresholdDistance = 0
# map_file = 'testingmap1.txt'
map_file = 'fuck6.txt'
segementedFile = True
numberOfFruits = 13


'''
##################################################################################################
##################################################################################################
                                        MAIN FUNCTION
##################################################################################################
##################################################################################################
'''
def main():

    #####################
    ## DEFINE VARIABLES
    ''' 
    List down the fruit list and fruit true position
    These is to be implemented to be obtained from true map directly eventually
    '''

    # Alphabetical Names of fruits
    fruits_list=  ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
    # fruits_list=  ['redapple', 'greenapple', 'orange']


    if segementedFile:
        positions = read_positions(map_file)
        aruco_true_pos = []
        fruits_true_pos = []
        # aruco_true_pos = np.zeros([10,2])
        # fruits_true_pos = np.zeros([5,2])

        for i in range(10):
            aruco_true_pos.append(positions[i])

        for i in range (10, numberOfFruits):
            fruits_true_pos.append(positions[i])

        aruco_true_pos = [list(pos) for pos in aruco_true_pos]
        fruits_true_pos = [list(fruit) for fruit in fruits_true_pos] 

    else: # go to default

        # True Position of fruits
        fruits_true_pos=  [[-0.4,  0  ],
                           [ 1.4, -0.8],
                           [ 0.8, -0.4],
                           [-1.2, -1.2],
                           [ 0.8,  0.8]]
        
        # True Position of Aruco Markers
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
    
    # Wanted Search List
    search_list =   ['orange', 'redapple']
    # search_list =   ['redapple', 'capsicum', 'orange']
    # search_list =   ['mango']
    search_index = []
    search_true_pos = []
    all_fruits_waypoints_list = []

    # Making a deep copy of fruits to be used in plotting and modifying obstacles
    fruits_copy = copy.deepcopy(fruits_true_pos)

    # Starting Position
    src_coord = [0,0]
    

    #####################
    # For Loop to append the index of each wanted SEARCH LIST FRUIT from the FRUIT LIST
    for i in range(len(search_list)):          ## The shopping list only, so 3
        for j in range(len(fruits_list)):      ## The full list at 5
            if search_list[i] == fruits_list[j]:
                search_index.append(j)
    
    # Appending the True Position of the SEARCH FRUITS
    for i in range(len(search_index)):
        search_true_pos.append(fruits_true_pos[search_index[i]])
    
    
    #####################
    # Main Function which runs to A* Search Algorithm Continuously
    for i in range(len(search_list)):

        '''
        A* works in grid but we are given coordinates. So, we need to convert all of the coordinates into the grid system.
        '''

        # if i == 0:
        #     # Call the Convert Coordinates to Grid Function to convert src_coord to src_grid
        src_grid = convert_coord_to_grid(src_coord)

        # Obtain the first destination coordinates from the search list
        dest_coord = search_true_pos[i]
        
        
        #     print("J value: ", j)
        # print("Final!:", dest_coord)

        # Call the Convert Coordinates to Grid Function to convert dest_coord to dest_grid
        dest_grid = convert_coord_to_grid(dest_coord)
        # print("Dest Grid: ", dest_grid)
        

        # Call the Modify Obstacles Function to modify the grid every new run. Setting the current destination as an obstacle and the next destination as not an obstacle
        grid = modify_obstacles(aruco_true_pos, search_index[i], fruits_true_pos)
        
        distances = []
        # print(search_index)
        col = fruits_true_pos[search_index[i]][0]
        row = fruits_true_pos[search_index[i]][1]
        for j in range(len(aruco_true_pos)):
            value = calculate_h_value(row, col, aruco_true_pos[j])
            distances.append(value)
            # print("Distance ", i, " ", value)
        
        count = 0
        for k in range(len(aruco_true_pos)):
            if distances[k] <= thresholdDistance:
                count += 1
        
        if count >= 1:
            dest_grid = modify_destinations(dest_grid, grid)
            # print("Came here!")
        else:
            # print("Else!")
            for l in range (len(dest_coord)):
            # print(dest_coord)
            # print(dest_coord[j])
                if dest_coord[l] < 0:
                    dest_coord[l] += how_far_from_fruits
                    
                else:
                    dest_coord[l] -= how_far_from_fruits
                dest_coord[l] = round(dest_coord[l], 2)
            dest_grid = convert_coord_to_grid(dest_coord)

        

        # Obtain the array of waypoints
        waypoints = a_star_search(grid, src_grid, dest_grid)

        ##################### PRINT STATEMENTS
        # print(waypoints)
        ##################### 

        # Calling the plot waypoints function to plot the waypoints on a graph for better visualisation
        if modificationFlag:
            modi_waypoints = modify_waypoints(waypoints, search_true_pos[i])
            waypoints_lists = [waypoint.tolist() for waypoint in modi_waypoints]
            # print(waypoints_lists)
        else:
            waypoints_lists = waypoints

        ##################### 
        # Simplifying the waypoints by calling a simplified path
        if plotFlag:
            new_waypoints = simplify_path(waypoints_lists)
            # new_waypoints = waypoints_lists
            all_fruits_waypoints_list.append(new_waypoints)
            plot_waypoints(new_waypoints)


        # Reassign the current destination coordinate as the new starting coordinate
        
        dest_coord = new_waypoints[-1]
        src_coord = dest_coord
        # src_grid = dest_grid
        # print("Next!")
    
    # Calling the plot function to fully plot the whole map out once all the paths are done
    plot_full_map(aruco_true_pos, fruits_copy)
    plt.show()

    return all_fruits_waypoints_list



'''
##################################################################################################
##################################################################################################
                                    READ TRUE MAP
##################################################################################################
##################################################################################################
'''
##################### 
# READ TRUE MAP
def read_positions(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Parse the string as JSON
    positions_dict = json.loads(data)
    
    # Extract the positions (x, y) into a list
    positions = [(item["x"], item["y"]) for item in positions_dict.values()]
    
    return positions



'''
##################################################################################################
##################################################################################################
                                    A* PLANNING FUNCTIONS
##################################################################################################
##################################################################################################
'''
##################### 
# Check if a cell is valid (within the grid)
def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

##################### 
# Check if a cell is unblocked
def is_unblocked(grid, row, col):
    return grid[row][col] == 1

##################### 
# Check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

##################### 
# Check if a cell is within range
def is_within_range(row, col, dest, threshold=3.5):
    distance = calculate_h_value(row, col, dest)
    # print("Distance is: ", distance)
    return distance <= threshold

##################### 
# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

def calculate_h_value_points(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return abs(x1-x2) + abs(y1-y2)

##################### 
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

    # Print the path
    if printingFlag:
        for i in path:
            print("->", i, end=" ")
            
        print()


    ##############################################33
    # PRINTING OF ACTUAL PATH
    path_coord = convert_grid_to_coord(path).tolist()

    # Print the path
    for i in path_coord:
        print("->", i, end=" ")
        
    print()

    return path


##################### 
##################### 

# def is_safe(grid, i, j, buffer_distance):
#     for dx in range(-buffer_distance, buffer_distance + 1):
#         for dy in range(-buffer_distance, buffer_distance + 1):
#             new_i = i + dx
#             new_j = j + dy
#             if is_valid(new_i, new_j) and not is_unblocked(grid, new_i, new_j):
#                 return False  # If any surrounding cell is blocked, it's not safe
#     return True

# Implement the A* search algorithm
def a_star_search(grid, src, dest):
    global plotFlag
    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        plotFlag = False
        return

    # Check if the source and destination are unblocked
    # if (not is_unblocked(grid, src[0], src[1])) or (not is_unblocked(grid, dest[0], dest[1])):
    if (not is_unblocked(grid, dest[0], dest[1])):
        print("Source or the destination is blocked")
        plotFlag = False
        # print(f"Source: {src}, Dest: {dest}")
        # print("src: ", is_unblocked(grid, src[0], src[1]))
        # print("dest: ", is_unblocked(grid, dest[0], dest[1]))
        print(f"Source blocked: {grid[src[0]][src[1]]}, Dest blocked: {grid[dest[0]][dest[1]]}")
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
                # if not (is_safe(grid, new_i, new_j, bufferDistance)):

                    
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):

                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    if printingFlag:
                        print("The destination cell is found")
                    
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)

                    ## TODO: Directly changing the path_grid to a path_coord
                    path_coord = convert_grid_to_coord(path)
                    
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

    
##################### 
# PATH SIMPLICATION
# intention is to cut down on number of waypoints required to travel 
def simplify_path(all_waypoints, threshold=0.4):
    new_path = []

    if printingTestFlag:
        print(all_waypoints)

    for j in range(len(all_waypoints)-1):
        # print(all_waypoints)
        if j == 0:  # add the first waypoint in the list 
            new_path.append(all_waypoints[j])
        else: 
            new_path.append(all_waypoints[j])
            
            # check for same xcoord
            if (new_path[-1][0] == new_path[-2][0]):
                if round(abs(new_path[-1][1] - new_path[-2][1]), 2) < threshold:
                    if all_waypoints[j+1][0] == new_path[-1][0]:
                        new_path.pop()
            
            # check for same ycoord
            elif (new_path[-1][1] == new_path[-2][1]):
                if round(abs(new_path[-1][1] - new_path[-2][1]), 2) < threshold:
                    if all_waypoints[j+1][1] == new_path[-1][1]:
                        new_path.pop()
            
            # check same diagonal/direction
            elif (round(all_waypoints[j+1][0] - all_waypoints[j][0],2) == round(all_waypoints[j][0] - all_waypoints[j-1][0],2) and round(all_waypoints[j+1][1] - all_waypoints[j][1],2) == round(all_waypoints[j][1] - all_waypoints[j-1][1],2)):
                if calculate_h_value_points(new_path[-1],new_path[-2]) < threshold:
                    new_path.pop()
    # add in final waypoint
    new_path.append(all_waypoints[-1])

    for i in new_path:
        print("->", i, end=" ")
        
    print()

    return new_path 
 


'''
##################################################################################################
##################################################################################################
                                GRID, COORDINATE MANIPULATION
##################################################################################################
##################################################################################################
'''

##################### 
# MODIFY OBSTABCLES
# Inputs:
#       aruco_true_pos
#       search_fruit_ind
#       fruits_true_pos
# Returns:
#       A ROW by COL grid

# Function:
# This is to modify the grid. Essentially. How it works is that a destination cannot be denoted as an obstacle. Instead, it must be taken as the destination.
# However, after we arrive at that destination, we have to turn it back into an obstacle, and make the new destination not an obstacle.
def modify_obstacles(aruco_true_pos, search_fruit_ind, fruits_true_pos):

    ##################### PRINTING STATEMENTS #####################
    if printingTestFlag:
        print("True Position of fruits in modify, at the start: ", fruits_true_pos)
    ###############################################################

    # Declaring Variables
    grid = np.ones([ROW, COL], dtype = int) # 1 MEANS NO OBSTACLE
    obstacle_grid_list  = []
    obstacles_coord_list = []

    # Always make a new copy from the original
    fruits_copy = copy.deepcopy(fruits_true_pos)

    # Popping the current fruit search index
    fruits_copy.pop(search_fruit_ind)

    # Concatenating the obstacles
    obstacles_coord_list = aruco_true_pos + fruits_copy

    # Convert the obstacle coordinates to a obstacle grid
    for i in range (len(obstacles_coord_list)):
        obstacle_grid = convert_coord_to_grid(obstacles_coord_list[i])
        # print("Obstacles: ", i, " ", obstacles_coord_list[i])
        obstacle_grid_list.append(obstacle_grid)

    # Indexing the exact ROW and COL, and asisgning it a 0.
    # 0 MEANS OBSTACLE IS THERE
    for i in range(len(obstacle_grid_list)):
        grid[obstacle_grid_list[i][0], obstacle_grid_list[i][1]] = 0
        # print("Grid Coordinates: ", [obstacle_grid_list[i][1], obstacle_grid_list[i][0]], "Is it unblocked? ", convert_to_coord([obstacle_grid_list[i][1], obstacle_grid_list[i][0]]))
        fldksjlkdfa = convert_to_coord([obstacle_grid_list[i][1], obstacle_grid_list[i][0]])
        # print("This is some nonesense: ", fldksjlkdfa)
        # print(fldksjlkdfa[1][1], fldksjlkdfa[0][0])
        # plt.scatter(fldksjlkdfa[1][1], fldksjlkdfa[0][0], s=10)
        # plt.grid(True)

    ##################### PRINTING STATEMENTS #####################
    if printingTestFlag:
        for i in range(ROW):
            for j in range(COL):
                # print(grid[i][j], end=' ')
                print(f"Y: {i} X: {j} Value: {grid[j][i]}")
            print()
    ###############################################################
    return grid


###############################################################
###############################################################
# CONVERT XY COORD TO GRID COORD
# Inputs:
#       coord_dest
# Returns:
#       a list of row_col_grid_indexing

# Function:
# This is to transpose the xy coordinates into a one-on-one with the grid spacing
def convert_coord_to_grid(xy_coordinates):
    global spacing
    global divisor
    # Create an array that runs from -1.6 to 1.6
    space = [round(i*divisor, 2) for i in range(-160, 170, spacing)]
    row_col_grid_indexing = []

    # Assign the equivalent grid representation of the xy coordinate
    for i in range(len(xy_coordinates)):
        for j in range(len(space)):
            if xy_coordinates[i] == space[j]:
                row_col_grid_indexing.append(j)
    return row_col_grid_indexing


###############################################################
###############################################################
# CONVERT GRID COORD TO XY_COORD
# Inputs:
#       path_grid
# Returns:
#       [path_coord]

# Function:
# This is to transpose the grid spacing into a one-on-one with the xy coordinates
def convert_grid_to_coord(path_grid):
    global spacing
    global divisor
    space = [round(i*divisor, 2) for i in range(-160, 170, spacing)]
    path_coord = np.zeros([len(path_grid), 2])

    # While running through 0 to 32
    for i in range(len(space)):

        # For all the Y values
        for j in range(len(path_grid)):

            # If the Y value matches the current space
            # The space is equivalent to a number between the range of -1.6 to 1.6
            if path_grid[j][1] == i:
                path_coord[j][1] = space[i]

            # For all the X values
            for k in range(len(path_grid)): # x

                # If the X value matches the current space
                # The space is equivalent to a number between the range of -1.6 to 1.6
                if path_grid[k][0] == i:
                    path_coord[k][0] = space[i]
                
    return path_coord


def convert_to_coord(coordinate):
    global spacing
    global divisor
    space = [round(i*divisor, 2) for i in range(-160, 170, spacing)]
    path_coord = np.zeros([len(coordinate), 2])

    # While running through 0 to 32
    for i in range(len(space)):

        # For all the Y values
        for j in range(len(coordinate)):

            # If the Y value matches the current space
            # The space is equivalent to a number between the range of -1.6 to 1.6
            if coordinate[j] == i:
                path_coord[j] = space[i]

            # # For all the X values
            # for k in range(len(coordinate)): # x

            #     # If the X value matches the current space
            #     # The space is equivalent to a number between the range of -1.6 to 1.6
            #     if coordinate[k] == i:
            #         path_coord[k] = space[i]
                
    return path_coord


'''
##################################################################################################
##################################################################################################
                                OBSTACLE AND WAYPOINT TOLERANCING
##################################################################################################
##################################################################################################
'''
def increase_tolerance(grid):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(ROW):
        for j in range(COL):
            if grid[i, j] == 0:
                for successor in directions:
                    if i == 0 and successor[1] < 0:
                        pass
                    elif i == ROW-1 and successor[1] > 0:
                        pass
                    elif 0 < i < ROW-1:
                        
                        if j == 0 and successor[0] < 0:
                            pass
                        elif j == COL-1 and successor[0] > 0:
                            pass
                        elif 0 < j < COL-1:
                            grid[i+successor[1], j+successor[0]] = 0
    
    return grid


def modify_destinations(dest_grid, grid):
    global how_far_from_fruits
    scale = how_far_from_fruits*10
    original_dest_grid = copy.deepcopy(dest_grid)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    scaled_directions = [(x * scale, y * scale) for x, y in directions]
    possible_destinations_list = []
    distances = []
    foundFlag = False

    # print ("Scales!:", scaled_directions)
    returning_dest_grid = original_dest_grid

    for dir in directions:
        new_i = returning_dest_grid[0] + dir[0]
        new_j = returning_dest_grid[1] + dir[1]

        if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j):
            possible_destinations_list.append([new_i, new_j])

    for coordinate in possible_destinations_list:
        distances.append(calculate_h_value(0, 0, coordinate))
    
    maxDistance = distances[0]
    maxIndex = 0

    for index in range(len(distances)):
        if distances[index] > maxDistance:
            maxDistance = distances[index]
            maxIndex = index
    
    returning_dest_grid = possible_destinations_list[maxIndex]
        # for dir in directions:
        #     new_i = returning_dest_grid[0] + dir[0]
        #     new_j = returning_dest_grid[1] + dir[1]

        #     if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j):
        #         returning_dest_grid = [new_i, new_j]
        #         foundFlag = True
        #         break

    return returning_dest_grid


def modify_waypoints(waypoints, current_fruit_position):
    modified_waypoints = []
    distances = []
    col = current_fruit_position[0]
    row = current_fruit_position[1]
    for i in range(len(waypoints)):
        value = calculate_h_value(row, col, waypoints[i])
        distances.append(value)
        # print("Distance ", i, " ", value)

    count = 0
    for i in range(len(waypoints)):
        if distances[i] > 0.09:
            modified_waypoints.append(waypoints[i])
        else:
            break
    
    return modified_waypoints


'''
##################################################################################################
##################################################################################################
                                    PLOTTING FUNCTIONS
##################################################################################################
##################################################################################################
'''
def plot_waypoints(waypoints):

    x_points = []
    y_points = []
    for i in range(len(waypoints)):
        x_points.append(waypoints[i][0])
        y_points.append(waypoints[i][1])
    
    plt.plot(x_points, y_points, marker = 'o') 
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])  
    plt.xticks(space); plt.yticks(space)
    for i, (x, y) in enumerate(zip(x_points, y_points), 1):
        plt.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
    plt.grid()


def plot_full_map(aruco_true_pos, fruits_copy):
    global impactRadiusSize

    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])  
    plt.xticks(space); plt.yticks(space)
    x_aruco = []
    y_aruco = []
    x_fruits = []
    y_fruits = []


    for i in range(len(aruco_true_pos)):
        x_aruco.append(aruco_true_pos[i][0])
        y_aruco.append(aruco_true_pos[i][1])
    
    plt.plot(x_aruco, y_aruco, 'ok') 
    # plt.scatter(x_aruco, y_aruco, s=impactRadiusSize, facecolors='none', edgecolors='r')

    # Plot empty circles around ArUco markers with radius in data units (impactRadiusSize)
    for (x, y) in zip(x_aruco, y_aruco):
        circle = plt.Circle((x, y), impactRadiusSize, edgecolor='r', facecolor='none')
        plt.gca().add_patch(circle)

    for i, (x, y) in enumerate(zip(x_aruco, y_aruco), 1):
        plt.annotate(f'aruco_{i}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')

    # fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]
    # fruit_colour = ["red", "cyan", "orange", "yellow", "green"]
    fruit_colour = ["red", "cyan", "orange"]

    for i in range(len(fruits_copy)):
        x_fruits.append(fruits_copy[i][0])
        y_fruits.append(fruits_copy[i][1])
    plt.scatter(x_fruits, y_fruits, c=fruit_colour, s=100)
    # plt.scatter(x_fruits, y_fruits, s=impactRadiusSize, facecolors='none', edgecolors='r')

    # Plot empty circles around fruits with radius in data units (impactRadiusSize)
    for (x, y) in zip(x_fruits, y_fruits):
        circle = plt.Circle((x, y), impactRadiusSize, edgecolor='r', facecolor='none')
        plt.gca().add_patch(circle)


    for i, (x, y) in enumerate(zip(x_fruits, y_fruits), 1):
        plt.annotate(f'{i+10}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
    
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()



'''
##################################################################################################
##################################################################################################
                                    READ THE TRUE MAP
##################################################################################################
##################################################################################################
'''
def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    # with open(fname, 'r') as f:
    #     try:
    #         gt_dict = json.load(f)                   
    #     except ValueError as e:
    #         with open(fname, 'r') as f:
    #             gt_dict = ast.literal_eval(f.readline())   
    #     fruit_list = []
    #     fruit_true_pos = []
    #     aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        # for key in gt_dict:
        #     x = np.round(gt_dict[key]['x'], 1)
        #     y = np.round(gt_dict[key]['y'], 1)

        #     if key.startswith('aruco'):
        #         if key.startswith('aruco10'):
        #             aruco_true_pos[9][0] = x
        #             aruco_true_pos[9][1] = y
        #         else:
        #             marker_id = int(key[5])
        #             aruco_true_pos[marker_id-1][0] = x
        #             aruco_true_pos[marker_id-1][1] = y
        #     else:
        #         fruit_list.append(key[:-2])
        #         if len(fruit_true_pos) == 0:
        #             fruit_true_pos = np.array([[x, y]])
        #         else:
        #             fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        # return fruit_list, fruit_true_pos, aruco_true_pos


'''
##################################################################################################
##################################################################################################
                                DEFAULT FUNCTION TO RUN
##################################################################################################
##################################################################################################
'''

if __name__ == "__main__":
    main()