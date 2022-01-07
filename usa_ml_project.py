from numpy import exp, array, random, dot
from google.colab import files
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


print('Enter Features file\n')
inputs = files.upload()
train = pd.read_csv(io.BytesIO(inputs['usa_features.csv']))

outputs = files.upload()    # upload the input feature file
test = pd.read_csv(io.BytesIO(outputs['usa_labels.csv'])) # # assign it to a variable

X = train.values
y=test.values



#plot data in 3D image

math1 = train['daily_deaths'].values
read1 = train['total_cases'].values
write1 = train['total_deaths'].values

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math1, read1, write1, color='#ef1234')
plt.show()




###
#sample way of cleaning the data is substract derivative of devide it by standard deviatoin
###
x = (X - X.mean()) / X.std() 

#########################################################################

#here a column of ones is added at the start of features file
#for the value of x0 so that to directly multiply it with weight matrix.

#########################################################################

m = x.shape[0]
ones =np.ones((m,1))  
x = np.concatenate((ones, x), axis=1)   # Nuw X with X0's =1

##################################################################################
#in next three lines of code
# here data is devided into 60% trainig data, 20% validation set and 20% test data.
###################################################################################

x1,x_test , y1, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
training_set_inputs,x_val , training_set_outputs, y_val = train_test_split(x1, y1, test_size=0.2, random_state=42)


class Linear_Regression():#creating lenear regression class
    def __init__(self): #defining a function for parameters initialization

        random.seed(1) # using to create same values, each time the code run

        self.synaptic_weights =np.random.rand(4,1) #initialize random values of parameters

# training/fitting the model
      
    def training(self, training_set_inputs, training_set_outputs, number_of_training_iterations):#create function for training
      self.loss = [] #creating an array for storing error
      n = np.float(x.shape[0]) # total number of samples
        
      for iteration in range(number_of_training_iterations): #create for loop for updating parameters
        output = (self.prediction(training_set_inputs)) # finding prediction for input values through calling prediction function 
        error = output - training_set_outputs #measuring error
        mse = (1/n) * np.sum((error**2)) # measuring mean squared error
        self.loss.append(mse) # appending mse
        m=training_set_outputs.size # total number of samples
        adjustment = -0.01*(1/m)*np.dot(training_set_inputs.T, error ) # finding the optimized values for updating parameters
        self.synaptic_weights = self.synaptic_weights + adjustment # adding the derivative term to parameters for optemization 


           


# for calculating the test loss use the below section

    def test(self, x_test,y_test):   # creating a function for measuring test error
      self.losst = [] # array for storing error
      n = np.float(x.shape[0]) # total number of samples
      for i in range(np.size(x_test)): #for loop for finding error for all inputs
        output_1 = (self.prediction(x_test)) #finding the predicted output
        error_1 = output_1 - y_test    # calculating error
        mse_1 = ((1/n) * np.sum((error_1**2)))  # calculating mean squared error
        self.losst.append(mse_1) # appending error



#this is the fucntion for prediction

    def prediction(self, inputs):           ###### creating function for prediction of outputs
        output = (dot(inputs, self.synaptic_weights)) # calculating output from inputs
        return  output



if __name__ == "__main__":

    #Intialise linear regression model.
    linear_Regression = Linear_Regression() # initializing the linear regression class

    
    print("Random starting pyrameter values are: ")                                         #printing initial parameter values
    print("Random starting theta_0 weight is: ",linear_Regression.synaptic_weights[0])
    print("Random starting theta_1 weight is: ",linear_Regression.synaptic_weights[1])
    print("Random starting theta_2 weight is: ",linear_Regression.synaptic_weights[2])
    print("Random starting theta_3 weight is: ",linear_Regression.synaptic_weights[3])

    linear_Regression.training(training_set_inputs, training_set_outputs, 1000)

    print("New synaptic weights after training are: ")
    print("New  weight of theta_0 after training : ",linear_Regression.synaptic_weights[0])
    print("New  weight of theta_1 after training : ",linear_Regression.synaptic_weights[1])
    print("New  weight of theta_2 after training : ",linear_Regression.synaptic_weights[2])
    print("New  weight of theta_3 after training : ",linear_Regression.synaptic_weights[3])
    
    print("Considering new situation [1, 599, 599,599] -> ?: ")
  
    print(linear_Regression.prediction(array([1,599,6260,66])))
  
    print(linear_Regression.test(x_test,y_test))
    

# as the features are very complex and are in large number and we used just a linear model for training thats why the loss is high/huge.
    
    plt.figure(1)
    plt.title('Loss values')
    plt.plot(linear_Regression.loss)
    plt.ylabel('training loss')
    plt.xlabel('epoch')

    plt.figure(2)
    plt.title('test loss Loss values')
    plt.plot(linear_Regression.losst[0:10])
    plt.ylabel('test loss')
    plt.xlabel('epoch')