# Object Detection Task README

## Overview
This README provides a detailed description of the object detection task, including dataset creation and annotation, data augmentation, data partitioning, and model selection. The task focuses on detecting books in images containing text in five different languages (Russian, Arabic, Chinese, English, and Spanish), with a special requirement of handling images with multiple languages.

## 1. Dataset Creation and Annotation
The dataset for this object detection task consists of images containing books in various languages. The following steps were taken to create and annotate the dataset:

- **Data Collection**: Images containing books in Russian, Arabic, Chinese, English, and Spanish were gathered from various sources, including books, websites, and public domain resources.

- **Page Division**: The collected images were divided into individual pages to create a more granular dataset. Each page represents a separate object of interest for detection.

- **Labeling**: Bounding boxes were manually annotated around each book in the images. Special attention was given to ensuring that the dataset has a balanced representation of each language class.

- **Multi-Language Images**: To satisfy the special requirement of handling images with multiple languages, a separate group of images was created. These images contain books with text in several languages within a single image.

## 2. Data Augmentation
Data augmentation techniques were applied to increase the diversity of the dataset and account for rotated images. The following steps were implemented:

- **Rotation**: A script was utilized to randomly rotate the images within a range of -30 to 30 degrees. This rotation range was based on the discussion in the Stack Overflow thread [link](https://stackoverflow.com/questions/52050792/can-object-detection-models-adapt-to-rotation). Rotation was only applied if the difference in degrees exceeded 10, ensuring significant variation.

- **Bounding Box Adjustment**: Bounding boxes were adjusted accordingly after image rotation to maintain accurate object annotations.

## 3. Data Partitioning
The dataset was split into three subsets for training, validation, and testing, respectively, with the following proportions:

- Training: 80% of the dataset
- Validation: 10% of the dataset
- Testing: 10% of the dataset

This partitioning ensures that the model is trained on a diverse set of data and evaluated on unseen examples.

## 4. Model Selection
The TensorFlow Object Detection API was employed for model training and evaluation. The chosen model architecture for this task is `ssd_mobilenet_v2/fpnlite_320x320`. Several considerations guided this choice:

- **Computational Efficiency**: This model is computationally efficient, making it suitable for training on a variety of hardware setups.

- **Effectiveness**: Despite its efficiency, the model has demonstrated good performance in object detection tasks.

- **Batch Size**: Training was conducted with a batch size of 16 to strike a balance between training speed and memory usage.

- **Tensorboard Monitoring**: Tensorboard was used to monitor the loss values and other training metrics over time, enabling effective model optimization.

<img width="568" alt="Screenshot 2023-09-18 at 02 11 37" src="https://github.com/kristinazk/LanguageIdentification/assets/90059525/801cdb9b-db6d-4bb6-a1ba-3fa8cf8f3d28">

By following this structured approach, we aim to create an accurate and robust object detection model capable of detecting books in images with text in multiple languages. This README provides an overview of the key steps taken to achieve this goal.

## 5. Model Inference

### Inference Results

The predictions of the model are shown below. As we can see, the model is capable of accurately predicting a rotated image, as well as it can accurately distinguish multiple languages in the same image and draw bounding boxes accordingly. However, the model is the least accurate while trying to detect more than 3 languages from an image. This could have been easily fixed by adding more training data containing this configuration of images.

![arabic](https://github.com/kristinazk/LanguageIdentification/assets/90059525/db760427-9f72-4bd1-babf-60ab9096782e)
![chinese](https://github.com/kristinazk/LanguageIdentification/assets/90059525/5c37f81d-ce04-4673-a97b-e7ab589681f8)
![English](https://github.com/kristinazk/LanguageIdentification/assets/90059525/1ce5e4ed-56bd-4459-bbb7-5379f7b01aa9)
![englishChinese](https://github.com/kristinazk/LanguageIdentification/assets/90059525/01eca250-269c-4160-895c-c1501824345b)
![MultipleLanguageDetection](https://github.com/kristinazk/LanguageIdentification/assets/90059525/21e2bf73-5b97-4625-91f4-c2fa8f0b7167)
![RotatedDetectioin](https://github.com/kristinazk/LanguageIdentification/assets/90059525/2e8566d5-78f8-4e4e-829b-22337d39b590)
![Spanish](https://github.com/kristinazk/LanguageIdentification/assets/90059525/b04bf0f1-f504-4cd6-bcde-517391069e02)

### Accuracy

To calculate the accuracy, a mAP scores of several thresholds was considered: 

<img width="373" alt="Screenshot 2023-09-18 at 15 53 08" src="https://github.com/kristinazk/LanguageIdentification/assets/90059525/5458c78a-4d2a-44ff-ae41-130485f79b65">
