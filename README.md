# Défi IA 2021 - INSA Toulouse - Team Salle104 

This GitHub repository contains all the necessary files to reproduce the results performed by the team Salle104 from INSA Toulouse (Adeline Cénac, Luc Dubreuil, Lucile Vallet and Jérôme Xiol) for the 5th edition of ["Défi IA"](https://www.kaggle.com/c/defi-ia-insa-toulouse/overview). 

This edition of the "Défi IA" pertained to NLP. The task was simple: assign the correct job category to a job description. This was a multi-class classification task with 28 classes to choose from. 

## Results

We performed our best result by using a [XLNet](https://huggingface.co/transformers/model_doc/xlnet.html) model with a macro F1-score around 0.825 which allows us to be ranked 5th among 78 teams. We made our Python code run on Kaggle Notebook and it took almost 8h50 to run. Our model is a transformer which needs to be trained on the given dataset, hence we needed to use GPU to accelerate the training: we deciced to use the GPU from Kaggle Notebook which allows 9 consecutive hours of execution time. 

## Environment

All the Python libraries required for this challenge are listed in the requirements.txt file.

To build a functional conda environment, execute the following lines:

```bash
conda create -n Salle104 python=3.7.6
conda activate Salle104
pip install -r requirements.txt
pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Nonetheless, we strongly recommend to have a NVIDIA GPU on your system and a driver installed to have a reasonable execution time.

## Usage 

In order to reproduce the results, you need to clone this GitHub repository and run the Python script train_and_predict.py with the following bash command: 

```bash
python train_and_predict.py
```

Running this script will create a CSV file of the predictions in the Results folder. 



