# SDA-Data-Science-Bootcamp-Final-Project
Author: Eatidal AlMutairi
# HR Analytics: Job Change of Data Scientists
**Predict who will move to a new job**

![Images](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/HR%20Analytics.png) 

## Business Problem
The situation set up by the data set is that a company which is active in the Big Data and Data Analysis space is offer in courses to some of its employees. The company is offer paid training to their employees. Fortunately, many employees have signed up for these paid job trainings. However, they have been running into the situation where upon finishing a course they end up switching companies. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment.
## Objective
As a Data Scientist, I could determine of these candidates are really wants to work for the company after training or looking for a new employment and create human understandible insights. The project can helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates.
#### look at the data:
Data contains 19158 Observation and 14 Features.
##### Features
* enrollee_id : Unique ID for candidate
* city: City code
* city_ development _index : Developement index of the city (scaled)
* gender: Gender of candidate
* relevent_experience: Relevant experience of candidate
* enrolled_university: Type of University course enrolled if any
* education_level: Education level of candidate
* major_discipline :Education major discipline of candidate
* experience: Candidate total experience in years
* company_size: No of employees in current employer's company
* company_type : Type of current employer
* lastnewjob: Difference in years between previous job and current job
* training_hours: training hours completed
* target: 0 – Not looking for job change, 1 – Looking for a job change

**Target** is (target)

Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.

### Data Visualization
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Target.png)
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Men%20vs%20Women%20Who%20will%20change%20job%20more!.gif)
* From the Above Two Plots , We can see there are approx 75% number of employees which are not looking for job change that of rows as '0' , Whereas 25% looking for a job change that of rows as '1' .
* So , probablity is higher that a candidate will not change job, but data is imbalance and this can be very problematic to our model if not handled because it make it to be skewed to the 0 class and may label most points as zero due to this effect.
* This Probelm can be solve using Over-Sampling or Under-Sampling.

Findings on gender:
* We have more male employees than female. Others are the least in number.
* For all the employees staying and working for company, 90.6% (10,209)male and only 8.1% (912)female.
* For all the employees looking for a job change 88.9% (3012)male and only 9.6% (326)female.
* 141 "others" will continue to work and where as 50 of them are looking for a change.
* If we look at the ratio then more women and "others" are looking for a change than men.

![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Relevent%20Experience.png)
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Relevent%20Experience.gif)
* We have more employees with relevant experience 72% and 10,831 of such employees are staying back. 2961 employees are looking for achange.
* All employees with no relevant experience 28% , 3550 will stay in job and 1816 are looking for change.
 
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Relationship%20between%20city%20and%20city_development_index.gif)
* Purple denotes employees who wants to stay and yellow denotes those who want a change.
* Lets put an imaginary line at 6.2 development index on y axis, Most of the yellow dots are below 6.2 index and purple dots above this line. We have more employees look for a change in cities with low development index.
* **City development index is an important factor who look for change.**

## Machine Learning
### Feature Selection
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Correlation%20between%20features.gif)
 
 From the above heatmap we can clearly observe that the target has a high dependance on the city_development_index which means candidates from city with higher amount of development index don't tend to change their jobs (corr is negative).
 
 According to my approach in selecting feature by 'SelectKBest' and 'f_classif' techniques, I Selected top 10 features from the data which is the best number that fit to my models.
 
 The 10 features that are selected are :
* city
* city_development_index
* gender
* relevent_experience
* enrolled_university
* education_level
* experience
* company_size
* company_type
* training_hours

### Apply Modeling
**Oversampling**
* The target feature is clearly imbalanced,The distribution of target has a lot more samples in '0' than in '1'.
* Used a Synthetic Minority Oversampling Technique (SMOTE) to increase data in balanced manner.
* The accuracy score using the RandomForestClassifier (Befor Oversampling) is : **0.78**
* The accuracy score using the RandomForestClassifier (After Oversampling) is : **0.86**

Data Balanced Successfully

Used Baseline model to compare the accuracy score (0.75) to three different machine learning models that applied on data as follows:

Random Forest Classifier Model(rfc)

              precision    recall  f1-score   support

         0.0       0.85      0.87      0.86      2877
         1.0       0.87      0.84      0.85      2876

    accuracy                           0.86      5753
    macro avg       0.86      0.86      0.86      5753
    weighted avg       0.86      0.86      0.86      5753

Decision Tree Classifier Model(dtc)

              precision    recall  f1-score   support

         0.0       0.80      0.80      0.80      2877
         1.0       0.80      0.80      0.80      2876

    accuracy                           0.80      5753
    macro avg       0.80      0.80      0.80      5753
    weighted avg       0.80      0.80      0.80      5753
   
KNeighbors Classifier Model(knn)

              precision    recall  f1-score   support

         0.0       0.88      0.66      0.76      2877
         1.0       0.73      0.91      0.81      2876

    accuracy                           0.79      5753
    macro avg       0.80      0.79      0.78      5753
    weighted avg       0.80      0.79      0.78      5753

### Evaluation the Model
My models has better accuracy scores than baseline model. That's good!

![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/Models%20Accuracy.png)

Although it is clear which model is more successfull, Look at its Confusion Matrix.

**Confusion Matrix**
This takes values and returns a report showing how each of the test values predicted classes compare to their actual classes.
![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/RFC-Confusion%20Matrix.png)
 
**ROC/AUC curve**
The Receiver Operating Characteristic (ROC) is a measure of a classifier’s predictive quality that compares and visualizes the tradeoff between the model’s sensitivity and specificity. When plotted, a ROC curve displays the true positive rate on the Y axis and the false positive rate on the X axis on both a global average and per-class basis.

![Image](https://github.com/eatidal/SDA-Data-Science-Final-Project/blob/main/Images/ROC%20%26%20AUC.png)

 Random Forest Classifier seem to be the most successful out of them
 
## Results
I have applied three different machine learning models to the data. Random Forest Classifier seem to be the most successful out of them. Random Forest achieved 0.86 accuracy score and 0.93 AUC scores. Analyzing the ROC curve which visualizes True Positivity Rate vs False Positive Rate for every threshold we can give the classifier,  It is perfect and I think we can call it a successful classification.

According to my approach in selecting feature importance we can see 'city_development_index', 'company_size' are most important factor in hob changes. Data scientists in cities with better development index which work in higher size of company don't tend to change their jobs.

**Resources**

[Code Examples](https://www.codegrepper.com/code-examples/python)

[Encoding Categorical Variables](https://kiwidamien.github.io/encoding-categorical-variables.html)

[Handling Imbalance](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/)

[oversampling](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

[Plotly](https://chart-studio.plotly.com/feed/#/)

**Data Source**

[Kaggle](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)

