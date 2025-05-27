# Neighbors for One target (N41)

We propose a novel approach termed "neighbors for one target (N41)," which improves anomaly detection accuracy for multivariate time-series data.
N41 redefines both the output and input of any detection model by "single target feature" and "its neighboring features," respectively.
Using a single target feature improves accuracy by allowing the model to focus solely on minimizing the training error (loss function) for that specific feature.
In contrast, defining multiple features as outputs distributes the training error across them, potentially leading to less effective error reduction for some features.
Incorporating neighboring features with patterns similar to the target feature further enhances accuracy by including relevant features.

This repository includes:  
(1) N41.py: An anomaly detection model built on a stacked GRU with N41  
(2) evaluation.ipynb: A script for evaluating results of N41 based models  
(3) similarity: A directory containing pairwise similarity values between features  
(4) evaluation: A directory containing functions for evaluations   


For evaluations, you should install the [eTaPR package](https://github.com/wshw4ng/eTaPR).
For testing the evluation, we share N41 results on the SWaT and HAI22.04 datasets as follows.
- [HAI22.04 results](https://drive.google.com/drive/folders/1VVBngdE8ubXYvvcRKxk08PjbzE2AJ-Fh?usp=sharing)
- [SWaT results](https://drive.google.com/drive/folders/1U5fpTYO4B6-_JRAq4gzAK8n4PvVSkaT4?usp=sharing)

My scripts are developed on Python3.9.13
You need the following packages: pytorch, numpy, pandas, math, argparse, time, datetime, pathlib, copy, ray, and open-cv.

## Anomaly Detection

### Dataset Preparation

Before running N41, you need to download the dataset files.
You can obtain the SWaT dataset by requesting access through the [official site](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
Also, the HAI22.04 dataset is available on the [Github repository](https://github.com/icsdataset/hai).

After preparing the datasets, update the following lines in N41.py to specify the paths to the downloaded dataset files.
```
# For SWaT dataset
# line 163-164

TRAIN_DATASET = ['~/Data/SWaT/SWaT_Dataset_Normal.csv']
TEST_DATASET = ['~/Data/SWaT/SWaT_Dataset_Attack.csv']


# For HAI dataset
# line 169-173

TRAIN_DATASET = []
for i in range(1, 7):
    TRAIN_DATASET.append('~/Data/hai-master/hai-22.04/train{}.csv'.format(i))
TEST_DATASET = []
for i in range(1, 5):
    TEST_DATASET.append('~/Data/hai-master/hai-22.04/test{}.csv'.format(i))
```


### Training and Testing

This script trains multiple models and computes scores for every timestep using the trained models.
- Arguments:
  - gamma: Controls the similarity strength between the target feature and its neighbors.
  - data: Specifies the dataset to train on - 'SWaT' or 'HAI22.04'.

Here is an example of execution:

```
python N41.py --gamma 0.3 --data hai22
```

If you want to run N41 with SWaT, you input "--data swat".
Gamma is set from 0.0 to 1.0.

As a result, the N41 script writes numpy files named after the index number of each target feature.
The files are stored in a directory called "result_{dataset name}".
For example, a file named "0.npy" contains anomaly scores from the model whose target feature is 0-th feature.


## Evaluation

It is important to consider the false positive rate (FPR) practically, but the FPR is overlooked in existing work.
In the evaluation, we restrict the maximum value of the FPR such as 0.001 or 0.002, which means 144 or 288 seconds of false alarms are allowed in a day (Actually, it is still generous constraint).
So, we select the best threshold (one or more) that satisfies the predefined FPR.

With the ensemble model, the threshold selection would be np-hard, which is resemble to the napsack problem.
So, we introduce three heuristic algorithms in the eval-util.py.
After running three algorithms, we choose the thresholds brings the best eTaF1 score.
All of these steps are executed with following function in evaluation.ipynb.
```
evaluate_ensemble_best(swat_label, np.array(swat_scores), 1e-3, False)
```

The result show a picture of detection results (a blue line) with label (a orange one) as follows.
![image](https://github.com/user-attachments/assets/3ed8fef9-96da-4507-ab51-00fcefb280c5)

Also, it presents the conventional F1, [point-adjust](https://dl.acm.org/doi/abs/10.1145/3178876.3185996), and [eTaPR](https://dl.acm.org/doi/10.1145/3477314.3507024).
![image](https://github.com/user-attachments/assets/d01174da-4b89-4de6-a560-7672b2fc3dbc)




