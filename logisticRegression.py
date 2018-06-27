import numpy as np

class LogisticRegression(object):

	def __init__(self, X, Y, alpha = 0.005, regularizer = 'l2', reg_coef = 0.01, steps = 500):
		self.Y = np.array(Y)
		self.alpha = alpha
		self.regularizer = regularizer
		self.reg_coef = reg_coef
		self.steps = steps

		#Standardize 
		x = np.array(X)
		self.X_mean = np.mean(x, axis = 0)
		self.X_std = np.std(x, axis = 0)

		#Add bias
		m, n = np.array(X).shape 
		bias = np.array([1]*m).reshape(m, 1)
		self.X = np.append(bias, (x- self.X_mean) / self.X_std, axis = 1)
		self.theta = np.array([0.0] * (n+1))

	#Sigmoid function
	def sigmoid(self, X):
		return 1.0 / (1.0 + np.exp((-1) * X))

	#Compute negative log likelihood
	def computeCost(self):
		h_theta_x = self.sigmoid(np.dot(self.X, self.theta))
		m, n = self.X.shape

		#Compute negative log likelihood
		neg_log_ll = -1.0 * sum(-1.0 * self.Y * np.log(h_theta_x) - (1.0 - self.Y) * np.log(1 - h_theta_x)) 

		#l1 and l2 regularization
		if(self.regularizer == 'l2'):
			neg_log_ll = neg_log_ll + 0.5 * self.reg_coef * sum(self.theta[1:] ** 2) / m
		else:
			neg_log_ll = neg_log_ll + 0.5 * self.reg_coef * sum(np.abs(self.theta[1:])) / m
			
		return neg_log_ll

	#Run gradient descent
	def gradientDescent(self):
		m, n = self.X.shape 

		#Compute gradient including regularization 
		for i in range(self.steps):
			curr_theta = self.theta
			h_theta_x = self.sigmoid(np.dot(self.X, self.theta))
			delta = self.Y - h_theta_x

			#Regularization term: 1/m 
			for j in range(1, n):
				self.theta[j] = curr_theta[j] - self.alpha * (1.0 / m) * (sum( -1.0 * delta * self.X[:, j]) + self.reg_coef * m * self.theta[j])

			print("Step %d : cost = %f" % (i, self.computeCost()))			

	#Predict labels given new X
	def predict(self, X):
		m, n = np.array(X).shape 
		x = np.array(X)
		bias = np.array([1]*m).reshape(m, 1)
		X = np.append(bias, (x - self.X_mean) / self.X_std, axis = 1)
		pred = self.sigmoid(np.dot(X, self.theta))

		#Get predictions based on 0.5 cutoff
		mask_upper = (pred >= 0.5)
		mask_lower = (pred < 0.5)
		pred[mask_upper] = 1
		pred[mask_lower] = 0

		return pred

