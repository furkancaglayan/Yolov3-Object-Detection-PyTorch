# Object Detection

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

## Useful Links
You Only Look Once: Unified, Real-Time Object Detection: https://pjreddie.com/darknet/yolo/

Darknet: https://pjreddie.com/darknet/

PyTorch Get Started: https://pytorch.org/get-started/locally/ 

Coco Dataset: https://cocodataset.org/
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
$ conda create --name test_env
``` 

This command will create a virtual conda environment named **test_env**. You can name your environment however you want. Now you must activate your newly created environment 
with below command.

```
$ conda activate test_env
```


### 3 - Install PyTorch

In order to install run and this project on GPU, first make sure you have a Cuda supported 
NVIDIA graphics driver and have the latest driver installed. This project is not tested on CPU. If everything is OK to this point and 
environment is activated, run below command to install PyTorch and torchvision on your machine.

For Windows: 
```
$ pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
```

For Linux: 
```
$ pip install torch torchvision
```

If installation completes without an error, proceed with the step 4.

### 4 - Requirements

To install other requirements, make sure you are at project folder in bash or 
Anaconda Prompt, depending on your operation system and then run following command to install 
all other requirements from the file 'requirements.txt'.

```
$ pip install -r requirements.txt
```

### 5 - Configuration Folder

Download the configuration folder from below link and extract it into the project workspace.

https://drive.google.com/file/d/1yFzclUCcUM4LhQhFbrzsjR5CLYETU7d1/view?usp=sharing

## How to Run?

To run the program on a local video, use the following syntax:

```
$ python detect.py video_path cfg_path weights_path names_path
```  

So an example usage would be:

```
$ python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names
```  
This command would detect objects in the video and displays the processed frames in a window. But,
it would not save the final video. To save the video you should explicitly specify the save flag and the
output path. Such as:

```
$ python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names --s -path det.mp4
```  
Above command would save the processed video at the end with the name 'det.mp4'. If you want to stop press space. program will terminate and
output until this point will be saved.

You can also set other optional parameters such as fps, confidence threshold and non-max 
suppression threshold.

```
$ python detect.py video.mp4 config/yolov3.cfg config/yolov3.weights config/coco.names --s -path det.mp4 -fps 30 -nms 0.5 -ct 0.9
```  

To get more information about parameters, run the following:

```
$ python detect.py -h
```  

### Run on Single Image

To run the detector on single image, use the same pattern of arguments. Except instead of video input, provide an image input with extensions
jpeg, jpg or png. And if you want to save the final image, provide a valid -path argument and set --s flag.

```
$ python detect.py image.jpg config/yolov3.cfg config/yolov3.weights config/coco.names --s -path image_det.jpg
```  


## Continue Development

When detect.py is called, program checks if inputs extension is png or jpg to validate if it's an image. if that is the case,
detection_callback() function is called with following parameters: image in Image(Pillow), darknet model, display flag and a color list.
Color list size is n where n is class count. Darknet model on the other hand is loaded at the start of the program with load_model() function.
It uses weights, cfg and names files. cfg file is used to initialize model while names file indicate unique classes which to predict.

If input is a video, then detect_from_video() function is called. This function generates frames from the video then calls the detection_callback()
function on each of them. However, this function can be customized. detect_from_video() has an argument called callback. So you can write your own
callback function with following parameter set: frame, darknet_model, display_video, color_list where display_video is used to show video on runtime. If set to false,
video will be processed to save without showing. 

To train a new detection model with custom dataset, (1) you have to create a new .names file with your classes, (2) annotate a new dataset with
tools such as CVAT or microsoft/vott so bounding boxes can be extracted. (3) split dataset into train and validation, (optional) download pretrained weights
to continue training on them, (5) create a new config file to your liking. At the end of the train you will have new weights. Now detect.py can be used with them.
A great tutorial exists in : https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e

      
  