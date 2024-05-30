# Person and PPE Detection using YOLOv8

This project implements a system to detect persons and their PPE (Personal Protective Equipment) using YOLOv8. The workflow involves converting annotations, cropping images, training detection models, and running inference to predict both persons and PPE items in images.

## Project Description

### Step 1: Annotation Conversion
We start by converting the PascalVOC annotations to YOLOv8 format using a custom script. This conversion is crucial for training YOLOv8 models, as it ensures the annotations are in the correct format.

### Step 2: Cropping Images
We use a script (`crop.py`) to crop images by detecting each person in the whole image. This step involves:
- Detecting persons in the whole image.
- Cropping the images to include only the detected persons.
- Using relevant annotations to draw appropriate bounding boxes on the cropped images for PPE detection.

### Step 3: Training the Models
We train two YOLOv8 object detection models:
- **Person Detection Model**: Trained using the `Filtered_dataset_Person` dataset and a `Person.yaml` configuration file.
- **PPE Detection Model**: Trained using the `Filtered_dataset_PPE` dataset and a `PPE.yaml` configuration file. This model detects various PPE items such as hard-hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses.

### Step 4: Running Inference
We create an `inference.py` script to run both models and predict persons and PPE in new images. The script processes a directory of images, applies the person detection model, crops the detected persons, and then applies the PPE detection model to the cropped images. The results, including bounding boxes and confidence scores, are saved to another directory.

## Usage


### Convert Annotations
Run the script to convert PascalVOC annotations to YOLOv8 format:
```sh
python pascalVOc_to_yolo.py
```

### Crop Images
Run the script to crop images and draw annotations:
```sh
python crop.py
```

### Train the Models
Train the person detection model:
```sh
yolo task=detect mode=train model=yolov8n.pt data=person.yaml epochs=100 imgsz=128
```
Train the PPE detection model:
```sh
yolo task=detect mode=train model=yolov8n.pt data=ppe.yaml epochs=100 imgsz=128
```

### Run Inference
Perform inference on a directory of images and save the results:
```sh
python scripts/inference.py --input-dir path/to/input/images --output-dir path/to/save/results person_model/dir ppe.model/dir
```
## Dataset

The dataset for this project can be downloaded from the following link:

[Download Dataset](https://drive.google.com/file/d/1myGjrJZSWPT6LYOshF9gfikyXaTCBUWb/view?usp=sharing)

Please unzip the dataset and place it in the `dataset` directory within the repository.

## Additional Notes

- The dataset is placed in the `dataset` directory of the repository.
- OpenCV's `cv2.rectangle()` and `cv2.putText()` are used for drawing bounding boxes and confidence scores instead of YOLOv8's built-in functions.
.
