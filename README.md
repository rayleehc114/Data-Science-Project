## Data Acquisition
To acquire the data, please use the link: https://www.kaggle.com/datasets/yasirumanujith/usa-real-estate-dataset
open the link and download the "csv" file to the "data" directory

Alternatively, you may download the data via google drive link: https://drive.google.com/file/d/1Xy-cbxdLzf9V9jo8YTTnKxJXMNSJxhLM/view?usp=share_link



## Installation
To run this project, please follow these stops in the terminal:

1. Clone the repository
    ```
    git clone <repository_url>
    ```

2. Acquire the data by downloading the csv file from the data acquisistion either link (same as above): 
    
    https://www.kaggle.com/datasets/yasirumanujith/usa-real-estate-dataset
    https://drive.google.com/file/d/1Xy-cbxdLzf9V9jo8YTTnKxJXMNSJxhLM/view?usp=share_link 
    
    then move the csv file into the "data" directory



3. Create and activate the conda environment:
    ```
    conda env create -f environment.yml
    conda activate D200_project
    ```

4. Install the package in development mode:
    ```
    pip install -e .
    ```

## Usage
1. Data preparation: Run the data cleaning and exploratory noteboook (`EDA.ipynb`) to generate the cleaned parquet file.
2. Model Training: Run the model training script (`model/model.py`) with code in termianl:
    ```
    python -m model.model
    ```


## For Your Information
The XGBoost model takes arounds 3 mins to train the data in Mac, which is the most among all three models.


## AI declaration
AI tools are mainly used in this project in plotting data distribution and model evaluation.