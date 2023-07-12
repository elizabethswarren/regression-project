# Zillow Model Project

# Project Description
This project aims to build a model using a variety of features to predict the value of homes.

# Project Goal
  * Determine what features in the dataset are related to home value.
  * Use these findings to create a model to predict the value of homes.
  * Share findings on the model with the Zillow data science team.

# Initial Thoughts
My initial thoughts are that property value will be dependent on the counties that they are located in since the home value is the tax assessed value. 

# The Plan
  * Acquire data
    
  * Prepare data
    * Rename the columns
    * Change the data types of certain columns
    * Rename the FIPS values to county for easier readability.
    * Drop the null values.
       * Nulls accounted for 0.15% of the data so I felt the best route was to simply drop them.
    * Drop outliers which skewed the data.
       * Outliers were determined using the IQR.
       * Values for all columns that fell outside of their respective fences were dropped.
    * Scale the data.
    * Split the data into train, validate, split in a 50, 30, 20 split.
      
  * Explore the data
    * Answer the questions:
      * Is the county the property located related to the value?
      * Is bedroom count related to home value?
      * Is bathroom count related to home value?
      * Are bedroom count and bathroom count related to one another? Can they be combined as a feature?
      * Is square footage of the property related to home value?
        
  * Develop a model to predict churn
    * Use RMSE as my evaluation metric.
    * Baseline will be the mean of home value.
   
  * Make conclusions.

# Data Dictionary
|**Feature**|**Description**|
|:-----------|:---------------|
|Bedroom Count | Numbers of bedrooms|
|Bathroom Count | Numbers of bathrooms|
|Square Footage | Total calculated square footage|
|Value | Assessed property value|
|County| County in California where property is located|

# Steps to Reproduce
  * Clone this repo
  * Acquire data
  * Put the data in the same file as cloned repo
  * Run the final_report notebook

# Conclusions
  * 

# Next Steps
  * 

# Recommendations
  * 
  
