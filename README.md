# GMM-Based-Color-Segmentation

This project was done as coursework for the course ESE650\-Learning in Robotics at University of Pennsylvania. It performs Color Segmentation using a gaussian mixture model from RGB data. The image pixels are segmented into 5 colors: Red, Black, Brown, Yellow, and Red (Not Barrel). The GMM model is then used to identify red barrels in the images using shape features and an estimate of the position of the barrel is obtained using a camera model trained simultaneously with GMM.

### Getting Started

The trained model has been generated and 'test.py' can be run directly with appropriate changes to the parameters. Training can be done again but the dataset needs to be fed in 'TrainingData/' folder.

##### Files

- train.py         : Independent Code. Reads the images in the folder 'Training_set'.
		   Run to generate training data that gets saved in the directory 'TrainingData' and subdirectories 
	           'Red', 'Black', 'Brown', 'Yellow' and 'Red_NoBarrel'. If any folder doesn't exist, it creates automatically.
		   After generating data, create a new directory, say 'Set' inside the directory 'TrainingData' and transfer the 5 color folders into it. 

- roipoly.py       : Utility function obtained from github. Allows to create polygons on displayed images.

- train_color.py   : Contains the function that trains the GMM model for all the colors using data obtained from train.py. 
		   Input folder is 'TrainingData/Set1'. Change the subdirectory appropriately in case data is retrained in train.py    
		   Takes the number of clusters per color as input. Saves the model as model_new.p 

- gmm.py		 : Contains the utility functions to compute the GMM model parameters using EM algorithm. Takes data and number of clusters as input

- train_camera.py  : Contains the function that computes and returns the camera parameters using least squares estimation. 
		   Takes the model generated in train_color.py or the (default) previously generated model 'model.p' and number of clusters as input 
		   Reads the images in the folder 'Training_set'

- test.py          : Contains the main test code. Uses the following parameters

##### Parameters

- folder = 'Test_set' : Folder in which the test images should be stored
- model_file = 'model.p' : Model to be loaded
- f = 0.55 : default focal length (calculated using least squares estimation)
- Train_Camera = False : Change to 'True' to train camera parameters real time using the previously generated GMM model
- Train_Color = False : Change to 'True' to train GMM model using previously generated data set
- Save_Output = False : Save the output images into the folder 'Test_output'. Creates if doesn't exist
- num_clusters = 3	: Number of clusters per colour. If being changed from 3 to any other number, then change the 'Train_Camera' and 'Train_Color' to True

During runtime, certain plots will be generated. They are the convergence plots of the log likelihood function. Kindly close them as they appear to proceed with the testing. Also close all the images as they appear during runtime to proceed with the running. The following output is a sample output that is produced for 1 test image.

> ImageNo = 1
> BottomLeft = (611.000000,466.000000)
> BottomRight = (650.000000,466.000000)
> TopLeft = (611.000000,406.000000)
> TopRight = (650.000000,406.000000)
> Centroid = (630.500000,436.000000)
> Width = 39.000000, Height = 60.000000, Distance = 10.655697
