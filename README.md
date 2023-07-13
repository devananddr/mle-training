# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To create conda environment
```
conda env create -f env.yml
conda activate mle-dev
```

## Install housing_project package

.whl file to install the Housing project package is located under dist directory. Initially navigate to dist directory by:
```
cd dist
```
Once entering the dist directory, install the .whl using the pip 

```
pip install housinglib-0.2-py3-none-any.whl
```
After installing the package, go back to the root dir by:
```
cd ..
```

## To excute the script
The script file are present in the script directory. Enter the script directory from root using:
 ```
cd scripts
```
To download and create training and validation datasets.
```
python ingest_data.py
```
To train the datasets use train.py 
```
python train.py
```
To see the performance of the model use score.py
```
python score.py 
```
Use -h argument to try various arguments available in the scripts.

