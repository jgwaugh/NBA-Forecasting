# Model Training 

Here, I train the LSTM. `model.py` contains the LSTM architecture
and `train.py` is the training script. 

Note that at this point, I don't tune hyperparameters. I may 
return to this at a later point in development. I have generally
found that hyperparameter tuning can have pretty marginal impact
on model outcome, although that does change with neural networks. I'd
want to tune network depth / number of parameters and maybe dropout rate. Additionally,
given the amount of compute resources I have available, complex hyperparameter
take a while. 


## Validation Loss 
![alt text](validation_loss.png)

Validation MSE decreases as a function of epoch, which 
is a promising sign that the model is indeed learning something. 
It also indicates that 30 epochs is roughly appropriate - 
the curve tends to flatten around epoch 20.