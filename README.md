# GFS-Final
SNU Master's Thesis Research
*** note that this repository is the source for my thesis paper

# GFS: 
These are codes of GFS (Graph-based Framework for Stock price movement prediction) used for predictiong future stock price movement, developed for my SNU Master's thesis research paper.
The title of the paper is 'Modeling Changing Stock Relations using Web Search Volume for Stock Price Movement Prediction'.
The paper is not yet public, so please note that this code is not open for references.

## Project Description
This project aims to predict the stock price movement of given stocks at time step t. The steps are as follows:
* Step 1: Collect historical price and web search volume for selected stocks. (Must be listed in the US market)
* Step 2: Preprocess data so that it can be used as input for the model
* Step 3: Build graphs according to the designated method
* Step 4: Make feature context vectors for each time series data (Price & Web search volume)
* Step 5: Use price context vectors as input for graphs, and send each graph through a GCN module
* Step 6: Using trend vectors and graph embeddings, make movement prediction


## Model
We provide no pretrained models for GFS in 'weights' directory. However, will be provided in the near future.

## Code Information
Codes in this directory are implemented using Python 3.7.
This repository contains the code for GFS. 
The required Python packages are described in ./requirments.txt.

* The major codes of GFS is in this directory.
    * `main.py`: the code that takes historical price and web search volume to perform stock price movement prediction.
    * `preprocess.py`: the code that collects needed stock related information and preprocess data so that it can be used in the main.py file.
    * `model/GFS.py`: the code that contains the implementation of GFS model.
    * `utils/trainer.py`: the code related to training of the model.
    * `utils/tester.py`: the code related to testing of the model.
    * `utils/get_price.py`: the code that collects price data.
    * * `utils/get_trend.py`: the code that collects trends (web search volume) data.
    * * `utils/load_graph.py`: the code that constructs graphs.

## How to use the code 

Type the following command to collect data:

```bash
    python preprocess.py --n (arbitrary dataset number) --start_date (YYYY-01-01) --mid_date (YYYY-07-01) -end_date (YYYY-12-31)
```

The script will create the following directory:
```
current directory
└── dataset_n
  └── Final
    └── Price
    └── Trend
    └── Keywords
```

To train the GFS model:

```bash
    python main.py --n (arbitrary dataset number) --start_date (YYYY-01-01) --mid_date (YYYY-07-01) --end_date (YYYY-12-31) --mode train --gpu (GPUs)
```

To test the GFS model:
```bash
    python main.py --n (arbitrary dataset number) --start_date (YYYY-01-01) --mid_date (YYYY-07-01) --end_date (YYYY-12-31) --mode test --gpu (GPUs) --best_model (best_model path)
```

IMPORTANT
To modify the list of stocks to be tested, please change the list of stocks contained in the stocks_info.csv file!!


