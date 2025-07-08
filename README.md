This is the working implementation of the course taught by Instructor Andrew NG, in collaboration 
of DeepLearning.AI with Stanford Online.

This implmentation uses anomaly detection algorithm on a diabetes.csv file, where we have taken two features
namely, the blood sugar level and Insulin Level.

The Algorithm marks anomalies that are outliers, and predict label y = 1, where the algorithm think that the
patient is dibetic, and 0, where it does not.

I have also included a Matplotlib Image where the outliers are labeled in red cross.
Also We can see that the algorithm has high recall, but it predicts 47 anomalies out of 46
