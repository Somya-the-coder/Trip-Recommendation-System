A Trip Recommendation System using:-
1.ML -> recommendation_engine.py
->data collection using web-scraping
->data preprocessing
  .Data Cleaning: This involves handling missing values, removing duplicates, correcting errors, and ensuring data consistency. 
  .Data Transformation: This includes scaling, normalization, encoding categorical variables, and feature engineering. 
    ..Scaling: Adjusting the range of numerical data to a specific scale (e.g., 0-1 or -1 to 1). 
    ..Normalization: Transforming data to have a specific distribution (e.g., zero mean and unit variance). 
    ..Encoding: Converting categorical data into a numerical format that machine learning algorithms can understand. 
    ..Feature Engineering: Creating new features from existing ones to improve model performance. 
  .Data Reduction: Selecting relevant features and reducing dimensionality to simplify analysis and improve model efficiency. 
  .Feature Selection: Identifying and selecting the most relevant features for analysis. 
  .Dimensionality Reduction: Techniques to reduce the number of variables while preserving important information. 
  .Data Integration: Combining data from multiple sources into a single, coherent dataset. 
  .Data Formatting: Ensuring consistent data types and structures. 
  .Data Validation: Checking for errors and ensuring the processed data meets the model's requirements. 
->content based filtering(new user)
->collaborative filtering(Hybrid for existing user)

2.Backend- Flask(python)- app.py[mainly for routing]

3.Frontend- vanilla javascript

4. Some python files to find confusion matrix of our KNN Model-> evaluation.py
