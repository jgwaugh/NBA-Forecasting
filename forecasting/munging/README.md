# Data Munging

`load_data.py` contains code that loads in the data from
`scraping` and stitches it together, preparing it for the LSTM. 

Note that I scale a copy of the data to lie in the [0, 1] range
in order avoid issues with different scales distorting L2 distances during
neural network training. 

Saved data from scraping on my machine are stored in `data`. 