# Model Training 

Here, I train the LSTM. `model.py` contains the LSTM architecture
and `train.py` is the training script. 

Note that at this point, I don't tune hyperparameters. I may 
return to this at a later point in development. I have generally
found that hyperparameter tuning can have pretty marginal impact
on model outcome, although that does change with neural networks. I'd
want to tune network depth / number of parameters and maybe dropout rate. Additionally,
given the amount of compute I have, complex hyperparameter
searches take prohibitively long. 


## Validation Loss 
![alt text](images/validation_loss.png)

Validation MSE decreases as a function of epoch which 
is a promising sign that the model is indeed learning something. 
It also indicates that 30 epochs is roughly appropriate - 
although more epochs could potentially be used.


## Forecasts

Use the streamlit app to validate model forecasts with
actual player performance - to do this, run 
`streamlit run app.py` in the terminal. You'll see something
that looks like the following:

![alt text](images/app_layout.png)

![alt text](images/klay_pts.png)


## Baseline Comparison

Additionally, I compare errors to a lagged baseline. I compare the average 
MSE between career predictions and true career 
outcomes, averaged over all seasons for a player and averaged over 
all players. 

At each timestep `K`, the model uses all `k < K` to make predictions,
competing against a simple lag. As shown below, the model 
 beats the lag. 

![alt text](images/baseline_error_comparison.png)


More rigorous baselines are always a good idea. That can come with time. 

