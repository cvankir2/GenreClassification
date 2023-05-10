## Part 4: Final Solution, Genre Classification Networks ##
Group Members: Camille Knott, Christine Van Kirk, Mia Manabat

# 1. Source Code: 

Source code for MLP solution is included in: MLP_BinaryClassification.ipynb
Source code for RNN solution is included in: RNN_Classification.ipynb
Source code for CNN solution is included in: CNNGenreClassification.ipynb

More detailed unfinished code and raw data found in Shared Drive at: https://drive.google.com/drive/u/0/folders/0AD62yCjloVIiUk9PVA 
	
## Instructions for running code
Run the code in Google Colab. Links to colab are in the ipynb files. 
  
# 2. Report:
## Description of Database:
We used the open source database at (https://github.com/mdeff/fma) and used both the small (8,000 tracks of 30 seconds – 7.2 GB) for all three networks. This dataset contained music from eight different genres, with an even distribution of the genres throughout the dataset. We shuffled and randomly split the dataset into 60% training, 20% validation, 20% testing categories. Due to the different natures of the models we had to preprocess the data for each type of network. 

### 2-Genre Classification MLP
For our 2-Genre Classification MLP model, we used the EchoNest audio features within the dataset to classify between Electronic and Rock music genres. The EchoNest features included attributes of acoustic-ness, danceability, energy, instrumentalness, liveness, speech-iness, tempo, and valence. This dataset is shown in ER_EchoNest_AudioFeatures.csv within our Github, and it contains balanced data for 4,000 electronic and rock music files.

<img width="601" alt="Screenshot 2023-05-09 at 8 14 50 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/13c9b1cc-e909-4394-91dd-97eb585e8e21">

### 8-Genre Classification RNN
For our 8-genre Classification RNN model, we used the 30 second mp3 files provided by the open source database. We converted them into MFCCs numpy files to be able to quantify the audio files and feed them into the RNN as a sequential array. The preprocessing involved includes using the Python audio processing module Librosa to convert the mp3 files directly into an array of values. The arrays are then kept, resized if possible (indexed), or discarded to create a dataset of the same sized arrays that would then be passed into the RNN. This preprocessing as well as the model creation are done in RN_Classification.ipynb located in the RNN folder, which is found in the main directory. The resulting dataset contains 7994 MFCC files which are then loaded again in the same file.

### 8-Genre Classification CNN
For our 8-Genre Classification CNN model, we used the 30 second mp4 music files and converted them into spectrogram pictures. This process involves converting the file to a wav file first and then into a spectrogram. This processing was done in MakeASpectrum.ipynb. The resulting dataset contains 3,425 spectrogram files. A new csv file was made with this smaller dataset for the ids and labels of these files in spectrograms.csv.



## Classification Accuracy:

### 2-Genre Classification MLP
Our final classification accuracy was 76% with 20 epochs and a 0.01 learning rate. 

### 8-Genre Classification RNN
Our final classification accuracy was 28.66% with 20 epochs and a .0001 learning rate. The accuracy of classifying the correct genre within the top 3 classifications was 60.75%.

### 8-Genre Classification CNN
Our final classification validation accuracy was 32.24%  on epoch 12 out of 20 with a 0.01 learning rate. We used the SGD optimizer with a momentum of 0.9. 
This resulted in a classification test accuracy of 37.23%. 



## Classification Analysis:

### 2-Genre Classification MLP
Although our 2-Genre Classification MLP performed well with a 76% testing accuracy, this is not as great as we would have hoped considering this model is only classifying between two genres. The distribution of values for most of the attributes of the EchoNest features varied in the range of (0, 1), which was possibly too sensitive for our network to appropriately separate and distinguish relative values. In addition, our final testing accuracy was roughly the same as our validation accuracy at the final stages of our network training; however, the testing accuracy is often lower because the testing dataset inherently includes data points that the network has not seen before. As such, the network is trained on the training/validation subsets and it is expected to see lower performance on unfamiliar data.


### 8-Genre Classification RNN
Model Summary:

<img width="536" alt="Screenshot 2023-05-09 at 8 14 38 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/ed52f06d-93f4-40bf-beb1-55b16bd2b096">

Figure 1. Summary of RNN Classification Model

The 8-Genre Classification RNN performed better than random chance (1/8 = 12.5%) at 28.66%. Although still better than random chance, there are many ways that the classification could have improved. As seen from the graph containing the training loss and validation loss in figure 1, the model severely overfitted to the training data. This would not be solved by increasing the number of epochs. Even after changing and adding layers in the model, adding dropout to reduce overfitting, and altering the learning rate, etc., the accuracy of the model did not change too much. As a result, the data that we are training on could be the cause of our accuracy. This could be explained by the size of the dataset that we have– in the future we could utilize the larger datasets of music samples contained in the open source database. With a smaller sample size, the model could easily learn these few samples as it runs through more epochs. Another cause of this accuracy could be the MFCCs that we created. MFCCs don’t contain everything about a song. If we created a more complex model with more data included, this may improve the accuracy. However, if the samples themselves were noisy, the accuracy that we can achieve could have a limit. This is different from the more qualitative features used in the MLP and more similar to the CNN classification. 
Overall the model was able to categorize the genre in the top 3 at a rate of 60.75%. Although the most likely genre was not always the correct classification, generally, the correct classification was usually in the top three likely genres.

<img width="432" alt="Screenshot 2023-05-09 at 8 14 29 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/2b642bd8-a4cb-47f1-b7f1-6a5a078c66dd">

Figure 2. Validation Loss vs. Training Loss of the RNN Classification Model during Training


<img width="434" alt="Screenshot 2023-05-09 at 8 14 22 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/54abd888-675c-462b-a4e5-2c404207541e">

Figure 3. Validation Accuracy vs. Training Accuracy of the RNN Classification Model during Training




### 8-Genre Classification CNN


<img width="363" alt="Screenshot 2023-05-09 at 8 13 57 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/97601c42-6a60-4e90-9e95-5f1341f95bab">

Figure 4. Summary of CNN Classification Model

The 8-Genre Classification CNN performed better than random chance (12.5%), the validation accuracy,  and the RNN model (28.66%) with a test accuracy of  37.23%. It is worth noting that in training the accuracy of the model did not increase after the 12th epoch. 
The accuracy of this model could be due to the spectrogram pictures. This could be due to the limited number of the spectrogram data pictures or the data the pictures show. The spectrograms themselves can provide a lot of information about the music; however, they do not always sufficiently cover all aspects of the pieces such as dynamics. This can cause the spectrograms to not be able to fully represent a piece of music. Different music genres could also be represented similarly with a CNN making it difficult for a CNN model to distinguish between them. 

<img width="390" alt="Screenshot 2023-05-09 at 8 13 35 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/1eaede85-cc91-4367-a2bf-13b90ad7b3d7">

Figure 5. Visualization of Feature Spaces with t-SNE on Test Data

Figure 5 illustrates the t-SNE visualization of the feature space below. There is definitely a bit of a distinction between certain genres of music Pop, Instrumental, Electronic, and Hip-Hop are for the most part separated. On the other hand, International music is much more spread out which might be in part due to the music incorporating different genres and the spectrograms not being able to identify language. Some genres, such as Experimental and Electronic, had a lot of overlap which makes sense as there is overlap in the styles and techniques between those genres. 
Figure 6, below, illustrates a similar t-SNE graph on the training data. This graph shows similar groupings to that of the tested data; however, International music has a more defined group towards middle of the graph. This might be due to the there being more International songs in the training set than in the testing set. This graph also shows the lower left corner of the graph being more varied in types of music. This might be some sort of overlapping quality that many of these pieces share.

<img width="459" alt="Screenshot 2023-05-09 at 8 12 55 PM" src="https://github.com/cvankir2/GenreClassification/assets/72172536/d0400346-322e-42b8-b511-fd17f117b279">

Figure 6. Visualization of Feature Spaces with t-SNE on Training Data

Conclusion:
	Of our models, the CNN performed best when classifying 8 genres with an accuracy of 37.23%; however, the difference was fairly minimal. We believe its level of accuracy is due to being able to differentiate certain genres fairly well, but not entirely accurate all the time. It is important to note that each model used different parts of the fma dataset so that the data would fit with the model. In this way, they each may be effective and recognize certain features of the music to differing levels of success. The next step in music genre classification would be figuring out how to combine these architectures to be more effective in handling the complexity and subjectivity of music genre classification.



# 3. Contributions

Camille:
* Downloaded data from source and uploaded to Google Collab
* Explored usage of CNN
* Tested MLP with all features of the data ~ still in progress		
* Developed the CNN 8-genre classification model

Christine:
* Curated balanced datasets of EchoNest audio features between different music genres
* Explored usage of MLP for binary genre classification and multi-genre classifications
* Expanded upon MLP from Practical 1 for binary classification of Rock/Electronic music data & quad-classification for Rock/Electronic/HipHop/Folk

Mia:
* Explored usage of RNN
* Created README file and report 
* Modified classification MLP code to use PyTorch's modules for cleaner code
* Developed the RNN 8-genre classification model

		
	


