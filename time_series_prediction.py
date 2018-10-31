
# importing the required libraries
import torch
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import pandas as pd
import numpy as np


def time_series_prediction(file_name,window_len, no_of_predictions,iterations):
	'''function to perform time-series prediction for the data in excel sheet format'''
	''' input: 1. filename - path to the excel sheet of data '''
	''' input: 2. window_len - number of samples in time-series data considered to make a prediction'''
	''' input: 3. no_of_predictions - the number of future predictions to be made '''
	''' input: 4. iterations - number of iterations for which the network has to be trained'''

	# read the data from excel sheet
	df_train = pd.read_excel(file_name, header=None)

	print ("start")
	def Design_matrix(df_train,window_len):
		'''function to split the time-series data into frames and stack each of these frames into a design matrix of dimensions number of frames x window length '''
		length_df_train = df_train.shape[0]
		Design_matrix = np.zeros((length_df_train- window_len,window_len))
		y_actual =np.array(df_train[window_len:]) # true label for the prediction is created from the data
		# splitting the data into frames
		for strt_index in range(length_df_train-window_len):
			Design_matrix[strt_index,:]= np.array(df_train[strt_index:strt_index+window_len]).reshape(1,window_len)	
		return Design_matrix,y_actual

	# obtaining design matrix and the target for the time-series data
	design_mat,y_actual = Design_matrix(df_train,window_len)
	
	# converting target and design matrix into variable
	torch_tensor_y_actual = Variable(torch.tensor(y_actual).float())
	torch_tensor_Design_mat = Variable(torch.tensor(design_mat).float())


	# initializing base class for neural netowk modules
	class Net(torch.nn.Module):
		def __init__(self):
			super(Net, self).__init__()       
			self.fc1 = torch.nn.Linear(design_mat.shape[1], 9)
			self.fc2 = torch.nn.Linear(9,1)
		def forward(self, x):
			x = F.relu(self.fc1(x))
			y = self.fc2(x)
			return y

	# the network created
	net = Net()
	

	# create a stochastic gradient descent optimizer with learning rate =0.00006 and momentum =0.6
	optimizer = torch.optim.SGD(net.parameters(), lr=0.00006, momentum=0.6)

	# create a mean squared error loss function
	criterion = torch.nn.MSELoss()

	
	for epoch in range(iterations):
			# setting the gradients in every iteration initially to zero
			optimizer.zero_grad()
			# input is passed through the network to generate predictions
			net_out = net(torch_tensor_Design_mat)
			# MSE loss is calculated between the prediction and the the true label
			loss = criterion(net_out, torch_tensor_y_actual)
			# the gradients calculated are back propagated
			loss.backward()
			# update the parameters using the gradients
			optimizer.step()

	# consider the last predicted value and the last row of design matrix to make future predictions by continuously appending the predicted value to the design matrix'''
	last_predicted_value =net_out[-1]
	last_row_Design_mat = torch_tensor_Design_mat[-1]

	
	# creating a list to store the predicted values
	predicted_values =[]

	for i in range(no_of_predictions):
		# initalizing a tensor to form the design matrix
		D_new = torch.zeros([1, window_len])
		D_new[0,0:-1] = last_row_Design_mat[1:]
		# appending the last predicted value into the new design matrix formed
		D_new[:,-1] = last_predicted_value
		# getting the predictions by passing the formed design matrix into the network
		pred_out = net(Variable(D_new))
		# appending the predicted value into list which is later returned to obtain all the predicted values
		predicted_values.append(pred_out)
		# returning the predicted value and the new design matrix formed for further predictions
		last_predicted_value = predicted_values[-1]
		last_row_Design_mat = D_new.float()
		last_row_Design_mat = last_row_Design_mat.view(window_len)
		
	return predicted_values


