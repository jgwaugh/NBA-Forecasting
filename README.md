# NBA-Forecasting

## Background 
This repo contains code from a 2019 independent project where
I looked to predict NBA player career using LSTM recurrent neural networks. 

Code was revisited and cleaned in January 2023. 

The basic idea is that a career is a time series of vectors 
`X_1, X_2, ...., X_N` where `X_j` is a `k` dimensional vector 
of points, rebounds, and other statistics that determine whether
a season was "good." My goal is then sequence prediction - 
given observations `X_1, ..., X_p` I want to predict
`X_p+1, ..., X_N` to determine how a career changes over time. The 
vector prediction nature of the problem lends itself to recurrent neural networks. 


## Setup

Modeling was done using `python 3.7.15` and virtual
environments were managed using `conda`.

Requirements are stored in `requirements.txt` 

I found it easiest to install deps with both `pip` (out of habit)
and `conda` (to circumvent long `tensforflow` compilation / builds from scratch)
. 

To install deps and build the environment, run `source install.sh`in a terminal shell. 

The repo is set up as a python package, to make imports between
modules easier. 

## Running

To make forecasts (after installing), activate your `environment`
with 
```
conda activate nba_forecasting
``` 

then run 

```
streamlit run app.py
```

The application should look something like this:

![alt text](images/app_top.png)
![alt text](images/app_bottom.png)

## Future Steps

Future work here should: 
1. Better handle aging of players
    1. Could be solved by adding longer lookback to batch generator in training
   2. Could also be solved by adding more hyperparameters
2. Better tune hyperparameters 
3. Add non parametric confidence intervals to the predictions. 