import os

# For the random images we've obtained online, this script renames the files to a certain format

def rename_all_files():

    # specify the directory path    
    currentDir = os.getcwd()        # GET CURRENT WORKING DIRECTORY
    # dir_path = os.path.join(currentDir,'cv/dataset/labels')   # ADD A SUBDIRECTORY
    dir_path = os.path.join(currentDir,'m3/arena')   # ADD A SUBDIRECTORY
    
    # dir_path = "/path/to/directory"
    # num = 1499
    # for num in range(0, 1499+1):
    # loop through all files in the directory
    num = 0
    for filename in os.listdir(dir_path):
    
    # check if the file is a text file
        if filename.endswith(".png"):

            # construct the old file path
            old_file_path = os.path.join(dir_path, filename)

            # construct the new file name
            new_file_name = f"arena_{num}.png"

            # construct the new file path
            new_file_path = os.path.join(dir_path, new_file_name)

            # use the rename() method to rename the directory
            os.rename(old_file_path, new_file_path)

            num+=1
            

if __name__ == '__main__':
    rename_all_files()