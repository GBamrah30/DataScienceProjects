The purpose of this ML project is to predict whether a Steam game will be successful using metadata available from the Steam API.

We will analyze what factors drive a game’s success and build a machine learning model to predict whether a game will be successful.

- We intiially did an exploratory analysis using one steam game API to investigate the type of information that the steam JSON files contain
- After our investigation, we developed a function that extracted the data of the top 2000 games with the specific data that we need
- We saved this data to a csv, created a new notebook and did further EDA as well as modelling of the data set
  
- Methodology:
Performed exploratory data analysis to visualize the relationship between release year and recommendations.
Transformed recommendations using a logarithmic scale to reduce skewness.
Fitted a linear regression model to quantify the effect of release year on recommendations.
Checked residuals to ensure model assumptions were met.

-Results:
The regression model shows a negative relationship between release year and recommendations: newer games tend to have fewer recommendations.
Specifically, the slope of the model indicates that each year newer, the expected number of recommendations decreases by ~21%.
Residuals were randomly scattered, indicating the linear model assumptions are reasonable.
The model’s R² is 0.186, meaning release year explains ~19% of the variation in recommendations. This is relatively low, which suggests other factors (like genre, popularity, and player count) also influence recommendations.
Overall, the analysis illustrates a clear trend over time, but more features would be needed for accurate prediction.

-Key Takeaways:
There is a clear decline in recommendations for newer games, likely reflecting cumulative exposure and community adoption over time.
Year alone is not sufficient to predict recommendations, highlighting the importance of additional features for predictive modeling.
