<hr>

<div align="center">

# Cat Image Classification with Deep Learning
  <a href="https://nisal.lk" target="_blank" rel="noopener noreferrer">
    <img src="https://kittyjump.vercel.app/api/kitty" width="350" height="200" style="border-radius: 15px;" alt="Visit Nisal.lk" />
  </a>
</div>
<br><br><br>


<div align="center">
<hr>

# Table of Contents 📑 
<br>
<hr>
</div>

- [Project Overview](#project-overview-)
- [Dependencies](#dependencies-)
- [File Structure](#file-structure-)
- [Usage](#usage-%EF%B8%8F)
  - [Step 1: Preprocessing Images](#step-1-preprocessing-images)
  - [Step 2: Training the Model](#step-2-training-the-model)
  - [Step 3: YOLO Detection (input validation)](#step-3-yolo-detection-input-validation)
- [Results](#results-)
- [License](#license-)


<hr>



## Project Overview 😺🛞

This project is designed to classify images of cats into two distinct categories:
1. **My Cat** 🐱
2. **Other Cats** 🐾

It leverages the power of deep learning and data augmentation to train a convolutional neural network (CNN) model that can accurately classify these images. The model uses a pre-trained ResNet-50 architecture, which is fine-tuned to work with the specific dataset of cat images.


<div align="center">

## ⭐⭐⭐UI⭐⭐⭐

</div>

1. **Initial Open** 🐱

<div align="center">
    <a href="https://nisal.lk/kitty-detector" target="_blank" rel="noopener noreferrer">
         <img src="https://res.cloudinary.com/dlnhogbjy/image/upload/v1738250416/isdatmyKitty_1_30_2025_8_46_21_PM_jqpzet.png"  />
    </a>
</div>

<hr>


2. **Image Upload and Getting the Results with the Custom Model** 🐱

<div align="center">
    <a href="https://nisal.lk/kitty-detector" target="_blank" rel="noopener noreferrer">
         <img src="https://res.cloudinary.com/dlnhogbjy/image/upload/v1738250755/isdatmyKitty_1_30_2025_8_46_36_PM_rxa6ct.png"  />
    </a>
</div>

<hr>


3. **False Input Identification with YOLO** 🐱

<div align="center">
    <a href="https://nisal.lk/kitty-detector" target="_blank" rel="noopener noreferrer">
         <img src="https://res.cloudinary.com/dlnhogbjy/image/upload/v1738250891/isdatmyKitty_1_30_2025_8_46_47_PM_jxkwgl.png"  />
    </a>
</div>

<hr>


4. **Accuracy Level is on Point** 🐱

<div align="center">
    <a href="https://nisal.lk/kitty-detector" target="_blank" rel="noopener noreferrer">
         <img src="https://res.cloudinary.com/dlnhogbjy/image/upload/v1738250879/isdatmyKitty_1_30_2025_8_46_58_PM_c1gidg.png"  />
    </a>
</div>

<hr>


5. **Harder To Find Images gave Great Results** 🐱

<div align="center">
    <a href="https://nisal.lk/kitty-detector" target="_blank" rel="noopener noreferrer">
         <img src="https://res.cloudinary.com/dlnhogbjy/image/upload/v1738251053/isdatmyKitty_1_30_2025_8_47_17_PM_ssj7bl.png"  />
    </a>
</div>

<br><br><br><br><br><br>
<hr>


## Dependencies 😸📦

To run this project, you need to install the following Python libraries:

- `torch`: For PyTorch and deep learning functionalities.
- `torchvision`: For model architectures and image transformations.
- `PIL` (Pillow): For image processing tasks.
- `tqdm`: For progress bars in training loops.
- `numpy`: For numerical operations.

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```
<br><br><br><br><br><br>
<hr>

## File Structure 🐱📖

The project directory is structured as follows:


### Project Folder

| **File/Folder**            | **Description**                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------|
| **models/**                 | Contains the pre-trained YOLOv5 model and any saved models for classification (`cat_classifier.pth`). |
| **run_file.py**             | The main script to run the web application for uploading images and displaying classification results. |
| **yolo_cat_detect.py**      | Handles YOLO model detection to verify if the image contains a cat before classification.         |
| **requirements.txt**        | A file listing all the dependencies (such as `torch`, `torchvision`, etc.) needed to run the project. |
| **README.md**               | Documentation for the project, including installation and usage instructions.                    |

### Scripts Folder

| **File**                    | **Description**                                                                                                      |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------|
| **preprocess_data.py**      | This script preprocesses the dataset, including image resizing, labeling, and formatting required for training.       |
| **train_model.py**          | The script used to train the model on the preprocessed dataset. It configures the model and starts the training process. |

<br><br><br><br><br><br>
<hr>

## Usage 🙀🗝️

To get started with the **isdatmyKitty** project, follow these steps:

### Step 1: Preprocessing Images

Before training or using the model, you'll need to ensure that the images are in the correct format. Follow these steps for preprocessing:

### Dataset Preparation

| **Step**                    | **Description**                                                                                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. **Prepare your dataset** | Gather images of your cat and other cats. Ensure these images are in formats like `.jpg`, `.png`, or `.jpeg`.                                                   |
| 2.  **Input your images**   | Put your cat images in `my_cat` folder and the other cat images in `other_cats` folder. This will help the model differentiate between your cat and other cats. |

<hr>

### Step 2: Training the Model

### Training Process

| **Step**                      | **Description**                                                                                                                                                                                                                                              |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. **Start training**         | The training code is designed to train a ResNet50 model to classify whether the image belongs to your cat or another cat. You can train the model using your labeled dataset. Ensure that your training script correctly references the preprocessed images. |
| 2. **Save the trained model** | Once the training is complete, save the model to a path specified in the code (for example, `models/cat_classifier.pth`). This model will later be used for classification during runtime in the Run File > `run_file.py`.                                   |


<hr>
<br>

### GPU-Accelerated Model -> How to Training with High-Resolution Images

### Overview

Training machine learning models with high-resolution images can be really demanding on your computer’s resources. But, using a GPU can make a huge difference. GPUs excel at handling complex computations and processing large amounts of data at the same time, which speeds up the training process. In this guide, we'll go over some important things to consider when using a GPU to train models with high-resolution images, and we'll share tips on how to make the most of your GPU depending on its specs.
<br><br><br>
### Key Considerations

| **Aspect**                | **Challenge**                                                                                   | **Optimization**                                                                                                  |
|---------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **GPU Memory (VRAM)**      | High-resolution images require more memory. Large images (e.g., 512x512 or 1024x1024) can quickly exhaust GPU memory, leading to out-of-memory (OOM) errors. | - Reduce image size (e.g., resize to 224x224) <br> - Use smaller batch sizes (e.g., 16 or 8) <br> - Monitor memory usage with `nvidia-smi` |
| **Image Resolution**       | Higher resolution images (e.g., 1080p or 4K) will consume more memory and slow down training. Large images lead to slower data loading times and longer training periods. | - Resize images to a smaller, fixed resolution (e.g., 224x224 or 256x256) <br> - Use data augmentation techniques (random rotation, flipping, cropping) |
| **Batch Size**             | High-resolution images, when combined with large batch sizes, may exceed the GPU’s memory capacity, leading to slowdowns or crashes. | - Begin with a smaller batch size (e.g., 8 or 16) <br> - Use gradient accumulation to simulate larger batch sizes |
| **Mixed Precision Training**| Training on high-resolution images can be computationally intensive, leading to slower training speeds. | - Use mixed-precision training (FP16) to reduce computational load and memory usage <br> - Use PyTorch's `torch.amp` with `GradScaler` |
| **GPU Utilization**        | High-resolution images can cause bottlenecks in GPU utilization if the batch size is too large or the images are too high resolution for the GPU's memory. | - Monitor GPU utilization with `nvidia-smi` <br> - If GPU utilization is low, increase batch sizes or use multi-GPU training |
| **Model Complexity (e.g., ResNet50)** | Complex models, such as ResNet50, can further increase memory and computation time when processing high-resolution images. | - Fine-tune deeper layers of the model and freeze earlier layers <br> - Use simpler models for faster experimentation |
| **Learning Rate Adjustment** | Training with large images may lead to a need for adjusting the learning rate to ensure stable convergence. | - Use learning rate scheduler (e.g., `CosineAnnealingLR`) <br> - Experiment with different learning rates for optimal performance |
| **Early Stopping**         | Training on high-resolution images can lead to overfitting, especially when using complex models or small datasets. | - Implement early stopping based on validation loss <br> - Regularly save the best-performing model based on validation metrics |
| **GPU Temperature Management** | Long training sessions with high-resolution images can cause the GPU to overheat, potentially throttling performance or damaging hardware. | - Monitor GPU temperature with `nvidia-smi` <br> - Reduce batch sizes or use smaller image resolutions if temperature is high |

<hr>
<br><br><br>

### Step 3: YOLO Detection (input validation)


| **Step**                       | **Description**                                                                                                                     |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Download YOLO model**        | The project uses the YOLOv5 model to detect if the image contains a cat before classifying it. The model is downloaded automatically if not already present in the `models` folder. |
| **Open the image**             | You can use the graphical interface to upload an image for validation. The app will check if a cat is detected using the YOLO model before proceeding to classification. If a cat is not detected, the application will notify you. |
| **Classify the image**         | After YOLO detects a cat, the model will classify whether it is your cat or another cat, based on the training. The confidence score will be displayed as a circle on the screen, with the confidence percentage and the classification result shown. |

<br><br><br><br><br><br>
<hr>


## Results 😼📊

The system is designed to detect whether an image contains a cat and classify whether the cat is your cat or another cat. The YOLOv5 model is used for detecting the presence of a cat, and a custom ResNet50 model is used to classify the detected cat based on the training.
<br><br><br>
### Testing Findings 🧪

| **Test Case**                        | **Expected Outcome**                                                  | **Actual Outcome**                                                                                                         | **Remarks**                                                                                                    |
|--------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Cat image of your pet**            | Model should classify the image as "my_cat"                          | Correctly classified as "my_cat" with a confidence of 95%.                                                                  | The model performed well with high accuracy for "my_cat" images.                                               |
| **Cat image of a different cat**     | Model should classify the image as "other_cats"                      | Correctly classified as "other_cats" with a confidence of 93%.                                                              | The model handled other cat images well.                                                                      |
| **Non-cat image (no cat detected)** | YOLO should detect no cat and provide an appropriate message.        | No cat detected, and the application displayed "No cat detected."                                                           | The YOLO model effectively prevented unnecessary classification attempts.                                      |
| **Cat image with low resolution**    | Model should still classify with an acceptable confidence level.     | The model showed reduced confidence (approx. 75%), but still correctly identified the cat.                                  | The model was able to handle lower-quality images but with less confidence.                                    |
| **Cat image with high occlusion**    | Model should classify correctly despite partial occlusion.           | The model classified the image as "my_cat" with a confidence of 85%, even with partial occlusion of the cat's face.        | The system was robust enough to classify images with partial occlusion, though the confidence was reduced.    |
<br><br>
### Model Performance

The YOLO model successfully detected cats in most cases, ensuring that the classification step was only triggered when relevant. The ResNet50 model achieved high accuracy on images with good quality and proper labeling. However, in cases of low-resolution images or heavy occlusion, the confidence was reduced, but the classification remained correct.
<br><br><br>
### Accuracy Metrics

| **Metric**                                      | **Accuracy**     |
|-------------------------------------------------|------------------|
| **Overall Classification Accuracy**             | >82%             |
| **YOLO Cat Detection Accuracy**                 | >98%             |
| **ResNet50 Classification Accuracy for "my_cat"**| >84%             |
| **ResNet50 Classification Accuracy for "other_cats"** | >90%         |
<br><br>


`These results indicate that the system is both reliable and accurate for classifying your cat vs. other cats under normal conditions. The accuracy might slightly drop when dealing with low-resolution images or partial occlusions, but the model is still capable of correct classification in such scenarios.
`



## License 😽😽

This project is licensed under the [MIT License](LICENSE).

### Copyright (c) 2025 Nisal Herath

<hr>



<div align="center">

`This repository is maintained by Nisal Herath. All rights reserved.`
<br>
`By using this code, you agree to the terms outlined in the LICENSE file.`


### [nisal@nisal.lk](mailto:anushka@nisal.lk)

### [nisal.lk](https://nisal.lk)
</div>
