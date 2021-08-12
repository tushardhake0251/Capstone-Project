# Capstone-Project
 
Product review categorization using multiclass classification refers to the assignment of suitable categories from a hierarchical category space to a document.  An AI model will be developed to measure the performance of our proposed multiclass classification method using the Amazon Review Dataset

1.	Industry Review

1.	Current Practices:
-Currently Amazon Machine Learning team can predict which Product’s review based on  
 Category with 95 percent accuracy.
-Product Review Categorisation can help the new player’s in the market to perform analysis on their products
-Amazon bet is that the future of work is one in which a machine understands the products based on reviews and category.
-According to the Consumer Technology Association’s Future of Work survey. Yet the tech industry is concerned that school systems and universities have not moved fast enough to adjust their curriculum to delve more into data science and machine learning. As a result, companies will struggle to fill jobs in software development, data analytics and engineering.

2.	Background Research:
-Now Amazon launched several new features for the Amazon SageMaker BlazingText algorithm. Many downstream natural language processing (NLP) tasks like sentiment analysis, named entity recognition, and machine translation require the text data to be converted into real-valued vectors. Customers have been using BlazingText’s highly optimized implementation of the Word2Vec algorithm, for learning these vectors from several hundreds of gigabytes of text documents. The resulting vectors capture the rich meaning and context that we recognize when we read a word. BlazingText being more than 20x faster than other popular alternatives like fastText and Gensim, enables customers to train these vectors on their own datasets containing billions of words using GPUs and multiple CPU machines, hence reducing the training time from days to minutes.

-Generating word representations by assigning a distinct vector to each word has certain limitations. It can’t deal with out-of-vocabulary (OOV) words, that is, words that have not been seen during training. Typically, such words are set to the unknown (UNK) token and are assigned the same vector, which is an ineffective choice if the number of OOV words is large. Moreover, these representations ignore the morphology of words, thus limiting the representational power, especially for morphologically rich languages, such as German, Turkish, and Finnish.
-Text classification is an important task in Natural Language Processing with many applications, such as web search, information retrieval, ranking, and document classification. The goal of text classification is to automatically classify the text documents into one or more defined categories, like spam detection, sentiment analysis, or user reviews categorization. Recently, models based on neural networks have become increasingly popular (Johnson et al., 2017, Conneau et al., 2016). Although these models achieve very good performance in practice, they tend to be relatively slow both at train and test time, which limits their use on very large datasets.
-To keep the right balance between scalability and accuracy, BlazingText implements the fastText text classification model, which can train on more than a billion words within ten minutes while achieving performance on par with the state of the art. BlazingText on Amazon SageMaker further extends this model by leveraging GPU acceleration using CUDA kernels, along with other add-ons like Early Stopping and Model Tuning, so that the users don’t have to worry about setting the right hyperparameters. 

3.	Literature Survey:

All Information in the world can be broadly classified into mainly two categories, facts and opinions. Facts are objective statements about entities and worldly events. On the other hand opinions are subjective statements that reflect people’s sentiments or perceptions about the entities and events . Maximum amount of existing research on text and information processing is focused on mining and getting the factual information from the text or information. Before we had WWW we were lacking a collection of opinion data, in which an individual needs to make a decision, he/she typically asks for opinions from friends and families. When an organization needs to find opinions of the general public about its products and services, it conducts surveys and focused groups. But after the growth of the Web, especially with the drastic growth of the user generated content on the Web, the world has changed and so has the methods of gaining one's opinion. One can post reviews of products at merchant sites and express views on almost anything in Internet forums, discussion groups, and blogs, which are collectively called the user generated content. As the technology of connectivity grew so as the ways of interpreting and processing of users' opinion information has changed. Some of the machine learning techniques like Naïve Bayes, Maximum Entropy and Support Vector Machines have been discussed in the paper. Extracting features from user opinion information is an emerging task.   
 

A generic model of feature extraction from opinion information is shown, firstly the information database is created, next POS tagging is done on the review, next the features are extracted using grammar rules such as adjective + noun or so on, as nouns are features and adjectives are sentiment words. Next Opinion words are extracted followed by its polarity identification. Some models also calculate sentence polarity for accuracy. Lastly the results are combined to obtain a summary. Many algorithms can be used in opinion mining such as Naive Bayes Classification, Probabilistic Machine Learning approach to classify the reviews as positive or negative, have been used to get the sentiment of opinions of different domains such as movie, Amazon reviews of products.  





2.	Data Dictionary and Pre-processing Data Analysis:
Range Index: 40000 entries (total 10 columns):

   Sr.No	Variables Names	Categorization of Variable	Null values Check
1.	ProductId	Categorical	40000 non_null object
2.	Title 	Categorical	39984 non_null object
3.	userId	Categorical	40000 non_null object
4.	Helpfulness	Numerical /Discrete	40000 non_null object
5.	Score	Categorical	40000 non_null object
6.	Time	Numerical /Discrete	40000 non_null object
7.	Text	Categorical	40000 non_null object
8.	Cat1	Categorical	40000 non_null object
9.	Cat2	Numerical /Discrete	40000 non_null object
10.	Cat3	Numerical /Discrete	40000 non_null object

In this dataset , we don’t have any null values in the dataset hence the dataset is free from null values.

















 


1.	 Data Attribute Details:
In the dataset, we will encode all the categorical values into Numerical Values as shown.

ProductId	ID of the Amazon Products
Title 	Title of the Products
userId	 Unique ID of the User
Helpfulness	
Score	 Rating of the product out of 5
Time	 It contains time
Text	 Reviews of all products.
Cat1	 It contains 6 sub-categories. 
 0(baby products)/1(beauty)/2(grocery gourmet food)/3(health personal care)/
4(pet supplies)/ 5(toys games).
Cat2	 It contains 64 sub-categories.
Cat3	   It contains 464 sub-categories.


2.	Irrelevant Columns:
There are some features which have irrelevant data or cannot contribute to the target variable. Hence we will drop them

time	The time records do not contribute to the text classification model hence we can drop it
score	This column will not contribute to our text classification model hence we can drop it.
userid	This column will not contribute to our text classification model hence we can drop it
helpfulness	This column will not contribute to our text classification model hence we can drop it

3.	Selecting the Most Important Features:

1.	Text
2.	Cat1
3.	Cat2
4.	Cat3


 

4.	Text PreProcessing:-


1.	Stopwords:-
           		 Stopwords are the most common words in any natural language. For the purpose of 
analyzing text data and building NLP models, these stopwords might not add much value to the meaning of the  document. Generally, the most common words used in a text are “the”, “is”, “in”, “for”, “where”, “when”, “to”, “at” etc.

2.	Lemmatization:- 
                     		Lemmatization is the process of grouping together the different inflected forms of a 
word so they can be analysed as a single item. Lemmatization is similar to stemming 
but it brings context to the words. So it links words with similar meaning to one word.
                     		Lemmatization are Text Normalization (or sometimes called Word Normalization) 
techniques in the field of NLP that are used to prepare text, words & documents for 
further processing.
                     
Lemmatization (or lemmation) in linguistics is the process of grouping together the 
inflected form of a words so they can be analyzed as a single item, identified by the 
word’s lemma, or dictionary form. In computational linguistics, lemmatization is the 
algorithmic process of determining the lemma of a word based on its intended meaning. Unlike stemming, lemmatization depends on correctly identifying the intended part of the speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document. As a result, developing efficient lemmatization algorithms is an open area of research.

For Example :
 
“Caring” >> Lemmatization >> “Care”


3.	Tokenization :-
                      		Tokenization is the process by which a large quantity of text is divided into smaller parts 
called tokens. These tokens are very useful for finding patterns and are considered as a 
base step for stemming and lemmatization. Tokenization also helps to substitute 
sensitive data elements with non-sensitive data elements.
               
Natural Language toolkit has very important module NLTK tokenize sentences which 
further comprises of sub-modules
1)	word tokenize
2)	sentence tokenize
Tokenization of words:- We use the method word_tokenize() to split a sentence into words. The output of word tokenization can be converted to Data Frame for better text understanding in machine learning applications. It can also be provided as input for further text cleaning steps such as punctuation removal, numeric character removal or stemming. Machine learning models need numeric data to be trained and make a prediction. Word tokenization becomes a crucial part of the text (string) to numeric data conversion. 
Example:-
from nltk.tokenize import word_tokenize
text = "God is Great! I won a lottery."
print(word_tokenize(text))
Output: ['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']
Tokenization of Sentences:- Sub-module available for the above is sent_tokenize. An obvious question in your mind would be why sentence tokenization is needed when we have the option of word tokenization. Imagine you need to count average words per sentence, how will you calculate? To accomplish such a task, you need both NLTK sentence tokenizer as well as NLTK word tokenizer to calculate the ratio. Such output serves as an important feature for machine training as the answer would be numeric. 
Example:- 
from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery."
print(sent_tokenize(text))
Output: ['God is Great!', 'I won a lottery ']
 			For the  “Text” column we had done Natural Language Processing (NLP) like 
Stopwords, Lemmatization,
 Tokenization  to make clean text.
 
                    
 


5. Project Justification:

●	This is a real data set of Amazon contains product review categorization data
●	The Data set is about Amazon Product Reviews analysis on Text.
●	This is a classification problem. The dependent variable is Cat1 for 1st approach.
●	We can use Classification model algorithms like Logistic Regression, Naive Bayes KNN, Decision Tree, Random Forest.
●	We can use bagging and boosting techniques after checking cross validation score for increasing the accuracy and performance of the model.



















 

3. Data Exploration (EDA) :-

1.	Relationship between the variables:
                  There are a total of 10  features, but we are using only two features for our analysis and model building 
 Text (Independent Feature)
Cat1(Dependent Feature)(Target) 
1.	Count of  Cat1 column :

![image](https://user-images.githubusercontent.com/76862211/129170665-a9e6ab99-9b62-49e8-b10d-61825b2c627e.png)
In our Cat1 column there are 6 sub-categories present, and those 6 categories have imbalance data.



2.	Count of Cat2 Column:
3.	
![image](https://user-images.githubusercontent.com/76862211/129170743-718d5329-bdbf-4933-b713-a78e7f03b669.png)

 

In our Cat2 column there are 64 sub-categories present, and those 64 categories have imbalance data.


3.	Count of Cat3 Column:
4.	
![image](https://user-images.githubusercontent.com/76862211/129170771-56723c44-6d4f-4e46-87d5-e10b2720fe29.png)


 

In our Cat3 column there are 464 sub-categories present, and those 464 categories have imbalance data.


2. Analyzing text statistics:
Look at the number of characters present in each sentence. This can give us a rough idea about the clean_text length:

![image](https://user-images.githubusercontent.com/76862211/129170295-0a4a1cf0-9509-4d16-b091-fe4776bf3766.png)


 


INFERENCE:- The histogram shows that clean_text range from 0 to 2000 characters and generally, it is  between 0 to 800 characters.

Now we can do data exploration at a word-level. Let’s plot the number of words appearing in each clean_text:

![image](https://user-images.githubusercontent.com/76862211/129170364-7aa087d4-f37a-45c9-a64f-9aa6e5db7ea5.png)


 


INFERENCE:- It is clear that the number of words in clean_text ranges from 0 to 300 and mostly falls between 0 to 110 words.
Let’s check the average word length in each sentence:

![image](https://user-images.githubusercontent.com/76862211/129170419-39eecc49-12b1-49cd-ac9d-821d6506251b.png)

 

INFERENCE:- The average word length ranges between 3 to 9 with 5 being the most common length. Does it mean that people were giving short word review


Analyzing the amount and the types of stopwords can give us some good insights into the data:

![image](https://user-images.githubusercontent.com/76862211/129170455-7407d331-f025-4465-b505-f06a8dccc3e8.png)



 
INFERENCE:-  We can evidently see that stopwords such as “the”, “and”, “to” are the most    dominate in the Text column.
So now we know which stopwords occur frequently in our text, let’s inspect which words other than these stopwords occur frequently.
We will use the counter function from the collections library to count and store the occurrences of each word in a list of tuples. This is a very useful function when we deal with word-level analysis in natural language processing:

![image](https://user-images.githubusercontent.com/76862211/129170538-98fe7ca8-b2c0-4b24-8cc8-a6c487bd652b.png)


 


INFERENCE:- The “use”, “get” , “one”, “like”& “love” are the words which are more occurring in the clean_text  column.

We will analyze the top bigrams in our clean_text:

![image](https://user-images.githubusercontent.com/76862211/129170571-5da17b88-e357-4a8c-b53a-a787644ac1f9.png)


 

INFERENCE:- We can observe that the bigrams such as ‘great product’, ’use product’ & ‘highly recommend’ that are related to product. 

 
 We will analyze the top trigrams in our clean_text:
 
![image](https://user-images.githubusercontent.com/76862211/129170603-5a278c01-9965-4836-a138-cdaf02f9072f.png)

 


INFERENCE:-  We can see that many of these trigrams are some combinations of “ highly recommended product” and “worth every penny”. 
3. Word Cloud: 
Word Cloud is a great way to represent text data. The size and color of each word that appears in the word cloud indicate it’s frequency or importance.
A word cloud (also known as a tag cloud) is a visual representation of words. Cloud creators are used to highlight popular words and phrases based on frequency and relevance. They provide you with quick and simple visual insights that can lead to more in-depth analyses.
Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. ... The dataset used for generating word cloud is collected from UCI Machine Learning Repository. It consists of YouTube comments on videos of popular artists.

●	For whole Cat1 Column:

![image](https://user-images.githubusercontent.com/76862211/129170874-bb168c93-1587-45c9-9399-51a811d1126d.png)



●	For Cat1 sub-Category-  (baby products)

![image](https://user-images.githubusercontent.com/76862211/129170907-dd61dc4b-5435-4867-ba39-03bcc0fd432c.png)





●	For Cat1 sub-Category-  (beauty)

![image](https://user-images.githubusercontent.com/76862211/129170934-a7675ed8-63da-4f52-a1b2-60097eb9f4c0.png)

 

●	For Cat1 sub-Category-  (grocery gourmet food)

![image](https://user-images.githubusercontent.com/76862211/129170962-49700843-811b-4683-b441-4cf4ba706b13.png)

 


●	For Cat1 sub-Category-  (health personal care)

![image](https://user-images.githubusercontent.com/76862211/129171004-a14ab91e-c1be-4ad1-a407-1ba992ce55a7.png)

 


●	For Cat1 sub-Category-  (pet supplies)

![image](https://user-images.githubusercontent.com/76862211/129171035-48d0a69d-5747-4033-9715-afaf324bc0ef.png)

 


●	For Cat1 sub-Category-  (toys games)

![image](https://user-images.githubusercontent.com/76862211/129171080-2ca509b6-dbf7-4ee8-b27f-433cb0bffbea.png)

 


4.	Imbalance Data Detection :-
In our dataset, the target feature is the  'Cat1' column & has six sub features.
We checked our ‘Cat1’ columns whether Imbalanced or not by using Count Plot.

 ![image](https://user-images.githubusercontent.com/76862211/129171117-d4b8f639-fe26-4f6d-a82d-d879ff2e6f74.png)


It Clearly states that the Data has an imbalance. toys games & health personal care Category has more counts than other. 


 
4. Approach 

Countvectorizer:-

CountVectorizer tokenizer(tokenization means breaking down a sentence or paragraph or any text into words) the text along with performing very basic preprocessing like removing the punctuation marks, converting all the words to lowercase, etc.
 The vocabulary of known words is formed which is also used for encoding unseen text later. An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appears in the document. Let's take an example to see how it works.
 The idea behind this method is straightforward, though very powerful. First, we define a fixed length vector where each entry corresponds to a word in our pre-defined dictionary of words. The size of the vector equals the size of the dictionary. Then, for representing a text using this vector, we count how many times each word of our dictionary appears in the text and we put this number in the corresponding vector entry.
For example, if our dictionary contains the words {MonkeyLearn, is, the, not, great}, and we want to vectorize the text “MonkeyLearn is great”, we would have the following vector: (1, 1, 0, 0, 1). 
 To improve this representation, you can use some more advanced techniques like removing stopwords, lemmatizing words, using n-grams or using tf-idf instead of counts.
The problem with this method is that it doesn’t capture the meaning of the text, or the context in which words appear, even when using n-grams.
Metrics Score using Countvectorizer:

Model/Matrix	Precision	Recall	F1-score	Roc-Auc Score
Logistic Regression	Train Scores	89.70	91.28	90.40	99.90
	Test Scores	82.85	83.81	83.31	97.11
Naive Bayes	Train Scores	87.33	70.03	74.33	97.71
	Test Scores	84.23	64.28	68.35	96.08
Decision Tree	Train Scores	99.87	99.85	99.86	99.99
	Test Scores	67.05	66.14	66.56	79.67
Random Forest	Train Scores	99.88	99.84	99.86	99.99
	Test Scores	82.16	76.03	78.32	95.53
KNN	Train Scores	68.12	65.74	64.86	92.81
	Test Scores	54.25	49.19	48.96	78.28


TF-IDF(Term Frequency Inverse Document Frequency) :-
                                                                                       
This is a technique to quantify a word in documents, we generally compute a weight to each word which signifies the importance of the word in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining.
TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)

Terminology
           t — term (word)
           d — document (set of words)
           N — count of corpus
           corpus — the total document set

TF-IDF for a word in a document is calculated by multiplying two different metrics:
●	The term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.
●	The inverse document frequency of the word across a set of documents. This means, how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
●	So, if the word is very common and appears in many documents, this number will approach 0. Otherwise, it will approach 1.
Multiplying these two numbers results in the TF-IDF score of a word in a document. The higher the score, the more relevant that word is in that particular document.
To put it in more formal mathematical terms, the TF-IDF score for the word t in the document d from the document set D is calculated as follows:
 
Where:
 
 

Metrics Score using TF-IDF :


Model/Matrix	Precision	Recall	F1-score	Roc-Auc Score
Logistic Regression	Train Scores	89.70	91.28	90.40	98.84
	Test Scores	82.82	83.78	83.30	97.11
Naive Bayes	Train Scores	87.33	70.03	74.33	97.71
	Test Scores	84.23	64.28	68.35	96.08
Decision Tree	Train Scores	99.87	99.85	99.86	99.99
	Test Scores	66.40	65.69	66.01	79.34
Random Forest	Train Scores	99.88	99.84	99.86	99.99
	Test Scores	81.93	76.02	78.28	95.42
KNN	Train Scores	80.41	57.85	49.56	99.07
	Test Scores	73.15	47.96	39.78	89.01



5. Assumptions
Regression:

Logistic Regression:
Logistic regression does not require a linear relationship between the dependent and independent variables. Second, the error terms (residuals) do not need to be normally distributed. Third, homoscedasticity is not required. Finally, the dependent variable  in logistic regression is not measured on an interval or ratio scale.
However, some other assumptions still apply.

First, binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.
Second, logistic regression requires the observations to be independent of each other. In  other words, the observations should not come from repeated measurements or matched data.
Third, logistic regression requires there to be little or no multicollinearity among the    independent variables. This means that the independent variables should not be too highly correlated with each other.

Fourth, logistic regression assumes linearity of independent variables and log odds. although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.
nally, logistic regression typically requires a large sample size. A general guideline is that you need at a minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).
 





Classification :

Decision Tree:
The below are the some of the assumptions we make while using Decision tree:
At the beginning, the whole training set is considered as the root.Feature values are preferred to be categorical. If the values are continuous then they are discretized prior to building the model.Records are distributed recursively on the basis of attribute values.

 


Random Forest:

No formal distributional assumptions, random forests are non-parametric and can thus handle skewed and multi-modal data as well as categorical data that are ordinal or non-ordinal


 

Naive Bayes:

Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.
 


KNN:

The kNN imputation method uses the kNN algorithm to search the entire data set for the k number of most similar cases, or neighbors, that show the same patterns as the row with missing data. An average of missing data variables was derived from the kNNs and used for each missing value (Batista and Monard, 2002). In the current study, k = 5, where the five closest individuals were used to impute missing data; this has previously been shown to be adequate (Hastie et al., 1999). One disadvantage of kNN is that an appropriate number of values from present data is necessary for each variable, which in the current study is five.
 




          
6. Treatment of balanced Data:

Random Under sampling Method:

![image](https://user-images.githubusercontent.com/76862211/129171388-cb1c0ebb-46b9-4630-8688-d1b3912c2986.png)


 Under sampling involves randomly selecting examples from the majority class to delete from the training dataset.
This has the effect of reducing the number of examples in the majority class in the transformed version of the training dataset. This process can be repeated until the desired class distribution is achieved, such as an equal number of examples for each class.
This approach may be more suitable for those datasets where there is a class imbalance although a sufficient number of examples in the minority class, such a useful model can be fit.
The scores obtained after Performing Random Under sampling are mentioned in the table below:
 
Model/Matrix	Precision	Recall	F1-score	Roc-Auc Score
Logistic Regression	Train Scores	91.35	91.41	91.34	99.10
	Test Scores	83.35	83.45	83.23	97.07
Naive Bayes Classifier	Train Scores	88.70	88.73	88.61	98.31
	Test Scores	84.22	84.41	84.15	97.21
Decision Tree Classifier	Train Scores	99.91	99.91	99.91	99.99
	Test Scores	69.79	69.69	69.67	81.85
Random Forest Classifier	Train Scores	99.91	99.91	99.91	100
	Test Scores	80.18	80.53	80.10	95.64
KNN Classifier	Train Scores	68.81	64.12	63.68	92.89
	Test Scores	55.87	49.62	49.62	79.04


The Scores observed in the above table after under sampling are similar to the Scores obtained using Countvectorizer method.

Data after Balancing :

 
i) Bagging:

While decision trees are one of the most easily interpretable models, they exhibit highly variable behavior. Consider a single training dataset that we randomly split into two parts. Now, let’s use each part to train a decision tree in order to obtain two models.
When we fit both these models, they would yield different results. Decision trees are said to be associated with high variance due to this behavior. Bagging or boosting aggregation helps to reduce the variance in any learner. Several decision trees which are generated in parallel, form the base learners of bagging technique. Data sampled with replacement is fed to these learners for training. The final prediction is the averaged output from all the learners.

ii) Boosting:

In boosting, the trees are built sequentially such that each subsequent tree aims to reduce the errors of the previous tree. Each tree learns from its predecessors and updates the residual errors. Hence, the tree that grows next in the sequence will learn from an updated version of the residuals.
The base learners in boosting are weak learners in which the bias is high, and the predictive power is just a tad better than random guessing. Each of these weak learners contributes some vital information for prediction, enabling the boosting technique to produce a strong learner by effectively combining these weak learners. The final strong learner brings down both the bias and the variance.
XgBoost:

XGBoost is an ensemble learning method. Sometimes, it may not be sufficient to rely upon the results of just one machine learning model. Ensemble learning offers a systematic solution to combine the predictive power of multiple learners. The resultant is a single model which gives the aggregated output from several models.

The models that form the ensemble, also known as base learners, could be either from the same learning algorithm or different learning algorithms. Bagging and boosting are two widely used ensemble learners. Though these two techniques can be used with several statistical models, the most predominant usage has been with decision trees.

After Bagging and Boosting, the following scores have been obtained:


Model/Matrix	Precision	Recall	F1-score	Roc-Auc Score
Logistic Regression	Train Scores	91.35	91.41	91.34	99.10
	Test Scores	83.35	83.45	83.23	97.07
Naive Bayes Classifier	Train Scores	88.70	88.73	88.61	98.31
	Test Scores	84.22	84.41	84.15	97.21
Decision Tree Classifier	Train Scores	99.91	99.91	99.91	99.99
	Test Scores	69.79	69.69	69.67	81.85
Random Forest Classifier	Train Scores	99.91	99.91	99.91	100
	Test Scores	80.18	80.53	80.10	95.64
KNN Classifier	Train Scores	68.81	64.12	63.68	92.89
	Test Scores	55.87	49.62	49.62	79.04
Bagging	Train Scores	99.88	99.88	99.80	99.99
	Test Scores	81.19	81.30	81.14	96.41
XGBoost	Train Scores	80.85	77.17	78.27	95.77
	Test Scores	79.37	76.00	76.98	94.86


KFold Cross Validation:

Cross-validation is a statistical method used to estimate the performance (or accuracy) of machine learning models. It is used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited. In cross-validation, there is a  fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.

![image](https://user-images.githubusercontent.com/76862211/129170008-c982b79c-4df0-4d99-a0ed-2a7debea7b96.png)


K-Fold Cross Validation Scores for All The Models:

 XGBC	Train Scores	Test Scores
Accuracy	77.29	76.16
Precision	80.85	79.37
Recall	77.17	76.01
F1_score	78.27	76.98
Roc_Auc score	95.77	94.86


The following boxplot below represents the scores of each model after K-fold Cross Validation:



Final Model:
  
 XGBC	Train Scores	Test Scores
Accuracy	77.29	76.16
Precision	80.85	79.37
Recall	77.17	76.01
F1_score	78.27	76.98
Roc_Auc score	95.77	94.86

When we observe train and test scores, XGBoost is performing well.
Considering this model as best model 
Limitation:

One of the biggest challenges is determining the length of strings to process in textual analysis. When textual data mining tools try to extract and analyze longer strings of characters, they are going to find fewer data points that meet their parameters. After doing preprocessing and count vectorizer, our text data was converted into vector form, so it is also a limitation for us.



1.	Different Imputation methods
2.	Cross validation
3.	Size of Data
4.	Hyper parameter tuning


 
	Implications

Preprocessing is one of the key components in a typical text classification framework. This paper aims to extensively examine the impact of preprocessing on text classification in terms of various aspects such as classification accuracy, text domain, text language, and dimension reduction.


Conclusion:

With the given dataset, we performed analysis using the various Data Visualization Techniques and applying some basic models like Logistic Regression, Decision Tree, Random Forest. We treated the imbalanced data and applied methods like Under Sampling, Hyper Tuning , Bagging and boosting and checked our scores. Since we were using NLP techniques to achieve categorisation of the text reviews, we used count vectorizer and TF-IDF techniques and analysed the scores for each of the following. We performed KFold Analysis as well for each of the models in order to get better accuracy.
Hence after vigorous analysis we decided to follow the count vectorizer approach and conclude that XGboost model is the best fit for our Business problem.

