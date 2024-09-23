# Creating a new Python environment

The following syntax includes creating a new python venv including ultralytics for access to YOLOv8 model


```python
python -m venv venv2
venv2\scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install ultralytics==8.0.196     
pip uninstall opencv-python
pip install opencv-contrib-python==4.6.0.66
```

## Commands

1. Command Prompt
```python
ssh pi@192.168.137.22
raspberrypi
cd ECE4078_MY
python listen.py
```

2. Lab folder
- Right click then type `cmd`

```python
venv2\scripts\activate
```

3. Operate

`python operate.py --ip 192.168.137.22`

As of M3, operate.py takes in the following arguments:

```python
--ip <robot ip address>
--port <defaults to 5000>
--calib_dir <defaults to calibration/param>
--yolov8 <.pt file for model weights, defaults to YOLOv8/best_10k.pt>
--map <txt file for true map, defaults to m3set1.txt>
```

4. Calibration

- Wheel Calibration
```python
cd calibration
python wheel_calibration.py --ip 192.168.137.22
```

- Camera Calibration
```python
cd calibration
python take_pic.py --ip 192.168.137.22
python camera_calibration.py
```


5. Running SLAM (M2)

- Generate Truemap: `python truemap.py <truemapfilename>`
    - note truemapfilename is WITHOUT the .txt extension
    - plot required coordinates then directly press X

- Robot operation
    - Run operate.py
    - Press ENTER to start SLAM and drive the robot around to look at ArUco markers
    - Press S to save map to slam.txt when done

- SLAM Evaluation: `python SLAM_eval.py <truemapfilename>.txt lab_output/slam.txt`


6. Object Detection (M3)

- Generate Truemap: `python truemap.py <truemapfilename>`
    - note truemapfilename is WITHOUT the .txt extension
    - plot required coordinates including that of the fruits then directly press X

- Robot operation
    - Call operate.py and include the truemap as an argument: eg. `python operate.py --ip 192.168.137.22 --map m3set2.txt`
    - Press ENTER to start SLAM, then press T to load true map
    - Drive robot around and look for fruits
    - Press P to run detection, and if results are satisfactory, press N to save image 
        - image is saved in pred_output folder as pred_n.png
        - current robot pose is saved to lab_output folder as an entry to pred.txt

- Object Pose Estimation: `python object_pose_estyolo.py`, inspect images and press any key to move on to the next image
    - once all images are done, object poses are saved to lab_output/objects.txt

- Object Detection Evaluation: `python cv_eval.py <truemapfilename>.txt lab_output/objects.txt`



7. Autonomous Navigation

- Waypoint Navigation




## Other libraries in requirements.txt

```python
machinevision-toolbox-python==0.5.5
numpy==1.23.1
requests==2.31.0
pygame==2.1.2
torch==1.12.1
torchvision==0.13.1
matplotlib==3.5.2
tqdm==4.64.0
h5py==3.7.0

```

Should any of the following libraries be of a different version or are simply unavailable,
call the following command (applies for both updates and fresh package installs)

```python
pip install <library name>==<version number>
```

# Current Libraries (Sandra's version)

```python
Package                      Version
---------------------------- --------------------
ansitable                    0.11.2
certifi                      2024.7.4
cfgv                         3.4.0
charset-normalizer           3.3.2
colorama                     0.4.6
colored                      2.2.4
cycler                       0.12.1
distlib                      0.3.8
filelock                     3.15.4
fonttools                    4.53.1
h5py                         3.7.0
identify                     2.6.0
idna                         3.8
kiwisolver                   1.4.5
machinevision-toolbox-python 0.5.5
matplotlib                   3.5.2
nodeenv                      1.9.1
numpy                        1.23.1
opencv-contrib-python        4.6.0.66
opencv-python                4.10.0.84
packaging                    24.1
pandas                       2.2.2
pillow                       10.4.0
pip                          24.2
platformdirs                 4.2.2
pre-commit                   3.8.0
psutil                       6.0.0
py-cpuinfo                   9.0.0
pygame                       2.5.2
pyparsing                    3.1.4
python-dateutil              2.9.0.post0
pytz                         2024.1
PyYAML                       6.0.2
requests                     2.31.0
scipy                        1.13.1
seaborn                      0.13.2
setuptools                   65.5.0
six                          1.16.0
spatialmath-python           1.1.11
thop                         0.1.1.post2209072238
torch                        1.12.1
torchvision                  0.13.1
tqdm                         4.64.0
typing_extensions            4.12.2
tzdata                       2024.1
ultralytics                  8.0.196
urllib3                      2.2.2
virtualenv                   20.26.3
```
