import json
import ast

def save_map(fname1="lab_output/slam_transformed.txt", fname2="lab_output/objects.txt", fname3="newmap.txt"):

    with open(fname1, 'r') as f:
        try:
            map = json.load(f)                   
        except ValueError as e:
            with open(fname1, 'r') as f:
                map = ast.literal_eval(f.readline()) 
    d = {}
    for key in map:
        d[key] = map[key]

    with open(fname2, 'r') as f:
        try:
            map = json.load(f)                   
        except ValueError as e:
            with open(fname2, 'r') as f:
                map = ast.literal_eval(f.readline()) 
    # d = {}
    for key in map:
        d[key] = map[key]
    
        
    with open(fname3, 'w') as f3:
        json.dump(d, f3, indent=4)

    print(f"New truemap saved as {fname3}!\n")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    parser.add_argument("--groundtruth", type=str, help="The ground truth file name.", default='truemap.txt')
    parser.add_argument("--estimate", type=str, help="The estimate file name.", default='lab_output/slam.txt')
    parser.add_argument("--transformed", type=str, help="The transformed file name.", default='lab_output/slam_transformed.txt')
    parser.add_argument("--combined", type=str, help="The combined estimate file name.", default='m5_map.txt')
    # NOTE changed this to parse known
    args, _ = parser.parse_known_args()


    # map can now combine slam est with fruit ests as a true map that can be read in :)
    save_map(fname1=args.transformed, fname3=args.combined)





