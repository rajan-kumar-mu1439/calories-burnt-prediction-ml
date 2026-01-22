<h1>Calories Burnt Prediction Using Machine Learning</h1>

👉Summary

Predict the number of calories burned during physical activity using user physiological and exercise data with machine learning regression models.

👉Overview

This project builds a machine learning regression system that estimates calories burned based on factors such as age, gender, height, weight, exercise duration, heart rate, and body temperature.


👉 Problem Statement

Manually estimating calories burned during workouts is inaccurate and varies across individuals.

Core Problems:

Generic fitness formulas ignore individual physiology

Over- or under-estimation affects fitness planning

Users need personalized calorie estimates

Solution:

Use machine learning regression models trained on real exercise data to predict calories burned more accurately.

👉 Dataset

Source :<a href="https://github.com/rajan-kumar-mu1439/calories-burnt-prediction-ml/blob/main/calories.csv"> Calories Dataset</a>

source :<a href="https://github.com/rajan-kumar-mu1439/calories-burnt-prediction-ml/blob/main/exercise.csv"> Exercise Dataset</a>

Key Features:Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp

Target Variable:Calories


👉 Tools & Technologies

Programming Language: Python

Libraries: NumPy, Pandas, Matplotlib / Seaborn, Scikit-learn

Modeling: Regression Algorithms

Environment: Jupyter Notebook

👉 Methods / Approach


 Data loading & inspection

 Data cleaning (missing values, duplicates)

 Encoding categorical features (Gender)

 Feature scaling (StandardScaler)

 Train-test split (80% / 20%)

 Model training (e.g., Random Forest / Linear Regression)

 Model evaluation (R² score, MAE)

👉 Key Insights

Duration and heart rate are the strongest predictors

Gender affects calorie burn patterns

Tree-based models outperform linear models

Feature scaling improves convergence and stability


👉 Dashboard / Model Output
<img width="1536" height="1000" alt="dashboard" src="https://github.com/user-attachments/assets/c85c5a5e-f221-4da8-a471-ab86569a9ac9" />

Sample Result:

Training Accuracy: 99%

Testing Accuracy: 99%

High accuracy is acceptable only if overfitting is checked — which this project does.

👉 Results & Conclusion

Successfully built a regression model to predict calories burned

Achieved high predictive accuracy on unseen data

Demonstrated end-to-end ML workflow

👉Conclusion:

Machine learning provides a more personalized and reliable way to estimate calories burned compared to static formulas.

🚀 Future Work

Deploy model using Streamlit / Flask

Add real-time wearable data

Improve generalization with larger datasets

Experiment with deep learning models

Integrate BMI & activity type


👤 Author & Contact

Author: Rajan Kumar

Role: Machine Learning Enthusiast

Email: rajankumarmu1439@gmail.com

LinkedIn: https://www.linkedin.com/in/rajan-kumar-mu1439/
