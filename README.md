# Social-Distancing-in-Real-Time
Social distancing in Real-Time using live video stream/IP camera in OpenCV. 

- Use case: counting the number of people in the stores/buildings/shopping malls etc., in real-time.
- Sending an alert to the staff if the people are way over the social distancing limits.
- Optimizing the real-time stream for better performance (with threading).
- Acts as a measure to tackle COVID-19.

---

## Table of Contents
* [Simple Theory](#simple-theory)
* [Running Inference](#running-inference)
* [Features](#features)
* [References](#references)

## Simple Theory
**Object detection:**
- We will be using YOLOv3, trained on COCO dataset for object detection.
- In general, single-stage detectors like YOLO tend to be less accurate than two-stage detectors (R-CNN) but are significantly faster.
- YOLO treats object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities.
- It is used to return the person prediction probability, bounding box coordinates for the detection, and the centroid of the person.

---
**Distance calculation:**
- NMS (Non-maxima suppression) is also used to reduce overlapping bounding boxes to only a single bounding box, thus representing the true detection of the object. Having overlapping boxes is not exactly practical and ideal, especially if we need to count the number of objects in an image.
- Euclidean distance is then computed between all pairs of the returned centroids. Simply, a centroid is the center of a bounding box.
- Based on these pairwise distances, we check to see if any two people are less than/close to 'N' pixels apart.

  ## Installation Steps:

  - Install latest Anaconda and open Anaconda Prompt
    
  - Run command
    '''
    conda activate tf
    '''
  - Install require softwares
    scikit-image ,scikit-learn, pandas, matplotlib, Pillow, plotly,
    opencv-python, spacy, lightgbm, mahotas, nltk, xgboost.
    
## Running Inference

- Install all the required Python dependencies:
```
pip install -r requirements.txt
```
- If you would like to use GPU, set ```USE_GPU = True``` in the config. options at 'mylib/config.py'.

- Note that you need to build OpenCV with CUDA (for an NVIDIA GPU) support first:


- Download the weights file from [**here**](https://drive.google.com/file/d/1O2zmGIIHLX8SGs24W7mjRyFKvE_CSY8n/view?usp=sharing) and place it in the 'yolo' folder.

- To run inference on a test video file, head into the directory/use the command:
```
python run.py -i mylib/videos/test.mp4
```
- To run inference on an IP camera, Setup your camera url in 'mylib/config.py':

```
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
url = ''
```
- Then run with the command:

```
python run.py
```
> Set url = 0 for webcam.

## References
***Main:***
- YOLOv3 paper: https://arxiv.org/pdf/1804.02767.pdf
- YOLO original paper: https://arxiv.org/abs/1506.02640
- YOLO TensorFlow implementation (darkflow): https://github.com/thtrieu/darkflow
