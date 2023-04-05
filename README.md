## Part 3: First Solution and validation performance ##
Group Members: Camille Knott, Christine Van Kirk, Mia Manabat

# 1. Source Code: 
	Source code is included in:
	

# 2. Report:
	The MLP is a good baseline for the music dataset because it is very flexible in working with the types of data we are attempting to use. For our baseline model, we used one hidden dimension with 90 neurons and the MSE function for the costs. The baseline dataset we used had 8 input dimensions for the music’s Echonest features and 1 output_dim. This is because we chose to start with a smaller subset of the genres of music to see if we would be able to do Binary Classification with the given data. We chose the Rock and Electronic genres for this since they were the most frequent genres in the data. This subset dataset has 4,000 data points. 

	Classification Accuracy: 75% 
	For 20 epochs at a .01 learning rate. 

For the final solution, we plan on implementing multiple different changes to this model. We would like to include more genres of music. Another reason we chose to classify a subset is because the classifier was not performing well (above 30%) on the larger dataset. For the final solution we would like to be able to classify more than two genres. 
The second addition to the final solution would be to include more input data on the music files. The dataset we are using includes more features we could use in classification. Of these we would like to include MCFFs in the final solution. 
Overall, we would like to add more layers, adjust their sizes, change the learning rate, and increase the amount of epochs to see if we can get a better performing model. 

# 3. Contributions

Camille:
a. Downloaded data from source and uploaded to Google Collab
b. Explored usage of CNN
c. Tested MLP with all features of the data ~ still in progress		

Christine:
a. Curated balanced datasets of EchoNest audio features between different music genres
b.  Explored usage of MLP for binary genre classification and multi-genre classifications
c. Expanded upon MLP from Practical 1 for binary classification of Rock/Electronic music data & quad-classification for Rock/Electronic/HipHop/Folk

Mia:
a. Explored usage of RNN
b. Created README file and report 

		
	



