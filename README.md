# Design, Development, Analysis and Performance Evaluation of Deep Learning Algorithms - Malaria Detection
# Introduction
Malaria is caused by parasites that are spread by mosquito bites. Malaria is a great burden on global health, with an estimated 200 million infections and 400,000 deaths per year. The majority of deaths occur in children in Africa, where a child dies from malaria almost every minute, and malaria is the main cause of childhood neuro-disability. Malaria symptoms include fever, exhaustion, headaches, convulsions, coma, and death in extreme situations  (Research, 2017). We can detect Malaria by examining the blood cell of patience with the syndrome. About 170 million blood films are examined every year for malaria, which involves the manual counting of parasites. The whole process is taken a very long time and requires professional training to be processed. Today, with the growth of deep learning applications in health care, we can build a classification system that can speed up the examining process which will save time, money and even people's life. 
Based on previous annotations from malaria field workers, we will build our own deep-learning model with the following process:
	- Explore the dataset
	- Build the input pipeline
	- Train the model
	- Evaluate the model
	- Tunning model and repeat whole process
# Methodology
## Exploration Data Analysis
In this project, we will use Malaria Cell Images Dataset (Arunava, 2020), the dataset has 2 folders: “Parasitized” (1), “Uninfected” (0), with a total of 27,558 images and a distribution of 50%:50% each class. The data is given in “.png” format and have 3 channels RGB with various image sizes. We will create a DataFrame with the following information: filename, weight, height, path, and class. Where filename is the image’s filename, weight and height are the image’s size, path is the path to read in the image, and class is its label: 1 for “Parasitized” and 0 for “Uninfected”.  The image size is various, distributed as in Figure 1

<img width="133" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/6f7f047e-902c-499b-a1a3-9ae9c3ec3827">

Figure 2 is our random sample from the dataset; from observation, we can see, blood cell with malaria is those cells with red point floating around with many shapes and colours that can be easily spotted by human eyes. 



<img width="179" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/08f6b39b-7669-40d5-80da-20f96503d46a">


Looking at the image sizes, we can get Table 1 with the width and the height mean of 132. The width and, height distribution is also close to normal as Figure 3 and Figure 4, it is reasonable to resize every image in the dataset to 130x130x3

Table 1: Image's size summary
|Width |Height|
|-|-|
|Mean|	132.49|	132.98|
|Mode|	130|	130|
|std|	20.02|	20.62|
|min|	46|	40|
|max|	394|	385|




<img width="216" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/e2c45cfe-2637-486a-b335-f9f64f84933f">


## Data Pre-process

Next, we will use Tensorflow’s image_dataset_from_directory which will load all the image in the folder corresponding to its class, resize the image to target image’s size (130x130x3 in this case) and splitting the dataset into training dataset and validation dataset with the proportion of 80:20 for training and validation respectively. The dataset is also batched into batch_size = 64. Notices, we need to set same set to split the data into train set and test set


<img width="226" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/973260fd-8935-4514-b0eb-6bd0c96115bd">


Before going into any Convolution Layers, we need to normalize every pixel into range 0 and 1. We will add a Rescaling Layer in the beginning of our model, it will be scaling every pixel by divide its value to 255 which is the maximum value for 8-bit RGB image. As we can see in Figure 5, the maximum, and the minimum value of image before and after passing through normalize layer.

## Model building and training
First, we will build the model from the combination of Convolution and MaxPooling layer. We created 3 continuous blocks of Conv and Pooling after rescaling layer, following by Flatten and 2 Dense layers as in Figure …. We choose the number of filters are 16, 32, 64 in the Conv layer respectively from the top of model to the end. We used padding=’same’ to avoid dimension reduction too fast, padding same are the parameter where Tensorflow will automatically add padding to our input so that the output of the convolution layer has the same size as input. And there is a Dense layer with size 128 before the classification layer and after the Flatten to slowly decrease the dimension size which will help the model learn more effectively.

<img width="226" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/5df4f86d-62ef-4026-80d0-c3ca5a63acf7">


<img width="161" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/c9b17d68-f26b-444f-8f9e-80bc3e222849">

We used Categorical Cross Entropy as our loss function because we only have 2 classes and number of samples each class are balanced, and Cross Entropy loss function is a popular choice for classification problems. For the optimizer, we used Adam optimizer with learning rate 0.001.

Cross Entropy= -∑〖y_i log⁡(p_i)〗

We train the model for 10 epochs with a EarlyStop as a callback, in this case, our setup for EarlyStop is monitor=’accuracy’, patience = 3: watch the accuracy of model during training, if the accuracy does not increase after 3 epochs, then stop training. The training process stop after 16 epochs.
## Evaluation
First, looking at the training loss, evaluation loss as well as training accuracy, validation accuracy in Figure 8 and Figure 9, the model can achieve very high accuracy in both training and testing, with a low loss, but the curve of training and testing are slowly diverged, it has a chance of getting overfitted if we train with more epochs. It is a sight of the model performing too good on the training set but not on the test set. Good model should have training and testing curve going down together and have very little differences in value. In this case, our model is learning the training dataset not the general of whole dataset, that is the reason why it continues to increase training accuracy, but the testing accuracy is keeping the same over training time.


<img width="117" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/42c39835-1766-40b7-8766-a1b938b1346e">


<img width="114" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/90945a86-2ffc-4274-b7aa-8532919e3957">



Because number of samples in each class are balanced, we will be using ROC curve for evaluation. Precision-Recall curve (PR curve) is only used when number in each class are imbalanced.
From above model, we get the following ROC (Figure 10 and Figure 11) 


<img width="104" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/030e89f5-5e4b-4a55-ba67-e6ed3456f6f1">


<img width="107" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/58a73113-6bfc-4e13-b107-354dabf255e8">



We can see that, the ROC curve in both classes is very good with AUC = 0.98 for both classes, it means the model perform good on the test set, the reason is that the dataset are very large and number of samples in each class are balanced. Looking at the classification report in Figure 10 we can observe that, the precision for the “Parasitized” is higher than “Uninfected”. The F1-score which is the balanced metric between precision and recall also achieve very high results (0.94), it means the model is good at classify positive class and negative class. 


<img width="192" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/3e5308a3-7f98-48b6-8ee9-3b351a990dc4">

## Tunning model
In this medical classification problem, it is very importance to detect the positive cases, because missing a positive case could leading to a fatal patience, we need to improve the model precision as high as possible. There are many ways to improve model precision and avoid overfitting:
- Increase training samples
- Add dropout
- Train with more epochs.
To increase training samples, there 2 ways: collect more sample, augment current data. We will use the second solution, add more data by augmenting the current training data. We use 3 augmented layers in Tensorflow: RandomCrop, RandomRotation, RandomZoom to randomly augment image in training dataset. We add the augment layer in front of the model, before rescaling layer. Random augmented images are shown in Figure 11 and Figure 12




<img width="66" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/b1cb8691-2b39-4628-958a-2b17d7d5c374">


<img width="77" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/69240308-8b75-4b7a-b9c8-a639a4c546f1">




In the first version of our model, because the model performed too good on the training set but worse on the testing set, we will try to make the model simpler in training and more complicate on testing. We will add dropout layer in front of the Flatten layer. During the training process, our model will randomly remove perceptron in Flatten and full layer will be used in the testing.
Lastly, we increased number of epochs used for training to 45 with EarlyStop callback, using Adam optimizer with learning_rate = 0.001. Figure 15 is our model summary with more layers: dropout, data augmentation.


<img width="156" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/aae8923f-7a2a-4983-b9c5-64806d9c2bc9">


Training process stopped after 16 epochs because the training accuracy does not go up after 3 continuous epochs. We get the following graph


<img width="118" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/096719fb-c76a-425d-a5ef-7731cb429693">


<img width="119" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/32bf784a-0e2c-441c-9a83-245144cbc40d">


 
Looking the graph after training, we can see the better performance model, the model has the average accuracy of 96% and the training and testing accuracy going upward together not diverged as earlier model. The model stopped after 29 epochs, no sight of being overfitted 
The ROC curve for both classes having AUC=0.99 increased 1% compared to previous model. They also have high True Positive Rate and low False Negative Rate.
For the classification report, the model has higher precision in “Parasitized”, the precision increased 3% compared to previous model. For the “Uninfected” class, the precision increased 2%, the average accuracy also increased 2% when compared to no augment, no dropout model.


<img width="120" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/2bd27a0d-90fc-42f1-935c-349507afc765">



<img width="111" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/a214266f-660b-467f-ae5f-31d0e2111ca7">



About the 2nd model confusion matrix in Figure 20, we are able to achieve 96% Precision and 0.96 F1-score.
<img width="452" alt="image" src="https://github.com/ntg2208/malaria_detection_CNN/assets/25520448/6182e966-b87c-425d-84f4-4eb60f954ef5">

Figure 20: 2nd model classification report
We experimented with the models by try to add augmented layers as well as Dropout or add both and get the results as in Table 2.

Table 2: Model tunning performances

|     Model version    |     Augment data    |     Dropout    |     Precision    |     Accuracy    |     F1      |     AUC     |
|----------------------|---------------------|----------------|------------------|-----------------|-------------|-------------|
|     model            |     No              |     No         |     0.94         |     0.94        |     0.94    |     0.98    |
|     model1           |     Yes             |     Yes        |     0.96         |     0.96        |     0.96    |     0.99    |
|     model2           |     No              |     Yes        |     0.96         |     0.96        |     0.96    |     0.98    |
|     model3           |     Yes             |     No         |     0.96         |     0.96        |     0.96    |     0.99    |

From table above, and from our focus metrics is model’s Precision, we can see that if we add more data or add dropout layer will increase model performance. There is a slightly difference when having data augmented return in higher AUC but it is only 1%.
For model performance, only have different of 2-3% is not much to consider changing the whole model architecture, but one of the importance is the training, validation loss, accuracy going approximately same, avoid overfitting in the future. It is importance to add Dropout before the Flatten layer.
# Conclusion
We build a model used for Malaria blood cell classification with light weight and high Precision ( 97%) thanks to the quality of the dataset. The model only uses 3 blocks of Conv and Pooling, combine with 2 fully connected layers but able to achieve very high Precision. With the advantage of model size, we can deploy the model to smaller edge devices such as mobile phone or cloud API which will help Malaria field workers to use the model easily, reduce examining time and money spend, help to save many children life in Africa as well as around the world. In the future for further improvement, we can increase the Precision by using Pretrained model or adding more data to our dataset.
 
# Bibliography
Arunava, 2020. Malaria Cell Images Dataset. [Online] 

Available at: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria?datasetId=87153&sortBy=voteCount
[Accessed 25 05 2022].

Research, R., 2017. Malaria Screener. [Online] 

Available at: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html
[Accessed 25 05 2022].

