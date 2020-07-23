# Object Detection

## Useful Links
You Only Look Once: Unified, Real-Time Object Detection: https://pjreddie.com/darknet/yolo/

Darknet: https://pjreddie.com/darknet/

PyTorch Get Started: https://pytorch.org/get-started/locally/ 

## TO-DO
- [x] Run object detection on local video
- [ ] Categorize vehicles according to their sizes.
- [ ] Object Detection on video stream.
- [ ] Object Detection on Unreal Engine 4
- [ ] (Optional) Velocity prediction of objects.
- [ ] (Optional) Depth estimation of objects.



## How to Install?

### 1 - Install Anaconda

To create a vonda environment first install Anaconda. To install Anaconda on Windows
follow the guides in [docs](https://docs.anaconda.com/anaconda/install/windows/). For Linux users,
a detailed [guide](https://docs.anaconda.com/anaconda/install/linux/) is present. After the installation,
proceed to the next step.

### 2 - Create Conda Environment

On Windows, find and open 'Anaconda Prompt(Anaconda3)' from the search menu. On Linux, open your bash. Write and 
run below command:
```
conda create --name test_env
``` 

This command will create a virtual conda environment named **test_env**. You can name your environment however you want. Now you must activate your newly created environment 
with below command.

```
conda activate test_env
```


### 3 - Install PyTorch

In order to install run and this project on GPU, first make sure you have a Cuda supported 
NVIDIA graphics driver and have the latest driver installed. This project is not tested on CPU. If everything is OK to this point and 
environment is activated, run below command to install PyTorch and torchvision on your machine.

For Windows: 
```
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
```

For Linux: 
```
pip install torch torchvision
```

If installation completes without an error, proceed with the step 4.

### 4 - Requirements

To install other requirements, make sure you are at project folder in bash or 
Anaconda Prompt, depending on your operation system and then run following command to install 
all other requirements from the file 'requirements.txt'.

```
pip install -r requirements.txt
```

## How to Run?

To run the program on a local video, use the following syntax:

```
python detect.py video_path cfg_path weights_path names_path
```  

So an example usage would be:

```
python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names
```  
This command would detect objects in the video and displays the processed frames in a window. But,
it would not save the final video. To save the video you should explicitly specify the save flag and the
output path. Such as:

```jupyter
python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names --s -path det.mp4
```  
Above command would save the processed video at the end with the name 'det.mp4'.

You can also set other optional parameters such as fps, confidence threshold and non-max 
suppression threshold.

```
python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names --s -path det.mp4 -fps 30 -nms 0.5 -ct 0.9
```  

To get more information about parameters, run the following:

```
python detect.py -h
```  

