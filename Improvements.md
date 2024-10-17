# Kirin Oleh Test

## Task 1

### Assigment
Create NER model for detecting mountains inside text.

### Target
Return tokens for sentence

### Dataset
I take dataset here [link](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset?resource=download), and [local](Task1/dataset/mountain_dataset_with_markup.csv)

### Solving
- Prepare train/eval dataset and normalize it [dataset](Task1/dataset/ready_dataset.pkl), [Python file](Task1/dataset.ipynb)
- Fine-tune pre-train Bert-NER model by 3 groups ["O", "B-LOC", "I-LOC"] and save it [here](Task1/model), [Python file](Task1/train.py)
- Inference model by ChatGPT dataset [here](Task1/inference.py)
- Demo model [here](Task1/demo.ipynb)

### Problem
I had problems training the model, because the dataset had a strong imbalance in sentences with keywords and those without keywords. In addition, there was an imbalance in tags, so I also adjusted the coefficients in the loss function, although it would be correct to calculate the exact coefficient by analyzing the data. And it is necessary to slightly adjust the metric for testing the model.

### Improvements
- Get more complex dataset for training
- Better adjustment of the lost function
- Add one-two layers on top of BERT (change model structure)
- Using CUDA torch for better performance

## Task 2

### Assigment 
Compare 2 image by keypoints

### Target
Return concat images with joint keypoints

### Dataset
The dataset that you provide doesn't match for this type of assigment, cause kaggle dataset provides from this [research](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9241044)

### Solving
- Use SuperPoint for keypoint detection
- Compare descriptors by cv2
- Return concat image

### Problem
It was really hard to me to create model for this task, because this need no linear dense on the start.
Also, that size of image too big for standard model training

### Improvements
- Use different model for keypoint detection such as [Keypoint-RCNN](https://pytorch.org/vision/main/models/keypoint_rcnn.html?ref=blog.roboflow.com), [CenterNet](https://github.com/xingyizhou/CenterNet?ref=blog.roboflow.com), [D2Net](https://github.com/mihaidusmanu/d2-net) or [YOLO](https://blog.roboflow.com/guide-to-yolo-models/) with fine-tuning
- Use algorithm for keypoint such as [SAR-SIFT](https://github.com/cfgnunes/sar-sift) or SIFT based
- Satellite models can be found [here](https://github.com/satellite-image-deep-learning/techniques)
- For image resizing we can use different methods from cv2, [docs](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
