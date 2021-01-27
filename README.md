# Défi IA 2021 - Team Salle104 

This GitHub deposit contains all the necessary files to reproduce the results performed by the team Salle104 (Adeline Cénac, Luc Dubreuil, Lucile Vallet and Jérôme Xiol) for the 5th edition of ["Défi IA"](https://www.kaggle.com/c/defi-ia-insa-toulouse/overview). 

## Results

This edition of the "Défi IA" pertained to NLP. The task was simple: assign the correct job category to a job description. This was a multi-class classification task with 28 classes to choose from. 

We performed our best result by using a [XLNet](https://huggingface.co/transformers/model_doc/xlnet.html) model with a macro F1-score around 0.825 which allows us to be ranked 5th among 78 teams. We made our Python code run on Kaggle Notebook and it took almost 8h50 to run. Our model is a neural network which needs to be trained on the given dataset, hence we needed to use GPU to accelerate the training : we deciced to use the GPU from Kaggle Notebook which allows 10 consecutive hours of execution time. 

## Installation of the libraries

All the Python libraries required for this challenge are listed in the requirements.txt file.

To build a functional conda environment execute the following lines:

```bash
conda create -n Salle104 python=3.8
conda activate Salle104
pip install -r requirements.txt
```

## Usage 

In order to reproduce the results, you need to clone this GitHub deposit and run the Python script blabla.py with the following bash command (from a terminal) : 

```bash
python blabla.py
```



