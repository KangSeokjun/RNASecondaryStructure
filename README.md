# Enforcing Constraints in RNA Secondary Structure Predictions: A Post-Processing Framework Based on the Assignment Problem

The source code for this work is designed to replace existing post-processing algorithms and improve performance. 
We applied the proposed post-processing method to the E2Efold and REDfold models as examples. 
This approach better satisfies the constraints of RNA structures, resulting in higher prediction performance and reduced running time. 
The source code can be found in the postprocess.py file, specifically in the postprocess_proposed function. 
This code was developed using the deep learning network from E2Efold and REDfold.

## Folder Structure

```
RNASecondaryStructure
├── E2Efold
│   ├── data # '.ct' or '.pickle' files for train
│   ├── data_generator
│   ├── data_preprocess # converting '.ct' to ',pickle' for train
│   ├── model
│   ├── train # train and test code
│   └── utils # postprocess and util functions
├── REDfold
│   ├── data # same folder structure as above
│   ...
└── README.md

```

## 1. E2Efold

### System Requirement
python(>=3.7.3)  
torch (>=1.2+cu92)  
pandas  
tqdm  
scipy  
numpy  

### Installation
The installation instructions are based on Anaconda:
```
conda create -n e2efold python=3.7.3
cd E2Efold
pip install -r requirements.txt
```
PyTorch must be installed to match the CUDA version.

### Data preprocessing
```
config.json:
{
  "ct_files_path": "../data/ct", // ct file path to make pickle file to train
  "length_limit": 600,  //
  "output_path": "../data/pickle/test.pickle",
  "save_path": "../data/pickle",
  "random_seed": 0
}
```

## 2. REDfold

### System Requirement
python(>=3.7.16)  
torch (>=1.9+cu111)  
biopython  
tqdm  
scipy  
numpy  

### Installation
The installation instructions are based on Anaconda:
```
conda create -n redfold python=3.7.16
cd REDfold
pip install -r requirements.txt
```
PyTorch must be installed to match the CUDA version.

## Reference
 
G. Suh, G. Hwang, S. Kang, D. Baek, and M. Kang, "Enforcing Constraints in RNA Secondary Structure Predictions: A Post-Processing Framework Based on the Assignment Problem," International Conference on Machine Learning (ICML), 2024