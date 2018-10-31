Function to perform time-series prediction for the data in excel sheet format using a two-layer fully connected network in PyTorch <br />
input: 1. filename - path to the excel sheet of data<br />
input: 2. window_len - number of samples in time-series data considered to make a prediction<br />
input: 3. no_of_predictions - the number of future predictions required<br />
input: 4. iterations - number of iterations for which the network has to be trained<br />
ouput: 1. predicted_values - list of predicted values<br />
