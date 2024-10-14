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