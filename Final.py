import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

"""
   Richard Lung 327084224
"""

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(color_codes=True)

# Load the dataset
df = pd.read_csv('original_wine_data.csv')


#==========================================================#
        #======== Basic Data Inspection ============#
#==========================================================#

print("\n======= Basic Data Inspection ========\n")
before = df #keeping copy of the before to compare the changes

# Specify number of rows and columns
rows, columns = df.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

# Display the first few rows and summary statistics
print("\n#===== Sample: =====#\n")
print(df.sample(n=15))

# Checking for all column types
print("\ntypes: \n", df.dtypes)


#========================================================#
       #============ Cleaning Data ============#
#========================================================#

#===== Checking for Errors, Nan|Null, Duplicate rows =====#
print("Number of missing values:\n", df.isnull().sum())
duplicate_rows_df = df[df.duplicated()]
print("Number of duplicate rows: \n", duplicate_rows_df.shape)

# No duplicate rows or Nan|Null values detected, proceeding with code


#========================================================#
  #============ Renaming and Reordering ============#
#========================================================#

print("\n======= Renaming and Reordering ========\n")

# Dropping 'Id' column, due to irrelevance.
df = df.drop(['Id'], axis=1)
print(df.head(5))
# No need for renaming, all names are valid and important to understand the wine compounds.


#========================================================#
     #============ Statistics Summary ============#
#========================================================#

print("\n======= Statistics Summary ========\n")
print(df.aggregate({
    'fixed acidity': ['mean', 'median', 'std', 'skew', 'kurt'],
    'volatile acidity': ['mean', 'median', 'std', 'skew', 'kurt'],
    'citric acid': ['mean', 'median', 'std', 'skew', 'kurt'],
    'residual sugar': ['mean', 'median', 'std', 'skew', 'kurt'],
    'chlorides': ['mean', 'median', "std", "skew", "kurt"],
    'free sulfur dioxide': ['mean', 'median', 'std', 'skew', 'kurt'],
    'total sulfur dioxide': ['mean', 'median', 'std', 'skew', 'kurt'],
    'density': ['mean', 'median', 'std', 'skew', 'kurt'],
    'pH': ['mean', 'median', 'std', 'skew', 'kurt'],
    'sulphates': ['mean', 'median', 'std', 'skew', 'kurt'],
    'alcohol': ['mean', 'median', 'std', 'skew', 'kurt'],
    'quality': ['mean', 'median', 'std', 'skew', 'kurt']}))

# Abnormal Skewness and Kurtosis in the following columns:
# residual sugar, chlorides and sulphates

#========================================================#
     #============ Detecting Outliers ============#
#========================================================#

# Due to abnormal observations of skewness and kurtosis in the statistics summary,
# We will try and detect outliers.

print("\n======= Detecting and Removal of Outliers ========\n")

# Calculate z-scores for numeric columns and remove outliers using z-score method
z_scores = np.abs(stats.zscore(df.select_dtypes(include=['number'])))
threshold = 3
before_df = df
df = df[(z_scores < threshold).all(axis=1)]

# Set up the plot
num_cols = df.select_dtypes(include=['number']).columns
n = len(num_cols)

plt.figure(figsize=(15, 5 * n))

for i, column in enumerate(num_cols):
    plt.subplot(n, 2, i * 2 + 1)
    sns.boxplot(x=before_df[column])
    plt.title(f'Before Outlier Removal: {column}')

    plt.subplot(n, 2, i * 2 + 2)
    sns.boxplot(x=df[column])
    plt.title(f'After Outlier Removal: {column}')

plt.tight_layout()
plt.show()

# Checking improvement on statistics
print(df.aggregate({
    'fixed acidity': ['mean', 'median', 'std', 'skew', 'kurt'],
    'volatile acidity': ['mean', 'median', 'std', 'skew', 'kurt'],
    'citric acid': ['mean', 'median', 'std', 'skew', 'kurt'],
    'residual sugar': ['mean', 'median', 'std', 'skew', 'kurt'],
    'chlorides': ['mean', 'median', "std", "skew", "kurt"],
    'free sulfur dioxide': ['mean', 'median', 'std', 'skew', 'kurt'],
    'total sulfur dioxide': ['mean', 'median', 'std', 'skew', 'kurt'],
    'density': ['mean', 'median', 'std', 'skew', 'kurt'],
    'pH': ['mean', 'median', 'std', 'skew', 'kurt'],
    'sulphates': ['mean', 'median', 'std', 'skew', 'kurt'],
    'alcohol': ['mean', 'median', 'std', 'skew', 'kurt'],
    'quality': ['mean', 'median', 'std', 'skew', 'kurt']}))

# By examining plots of after outlier removal, no invalid or extreme values detected


#========================================================#
    #=== Visualization of Effects on Wine Quality ===#
#========================================================#


#========= Graph Plots =========#
# List of parameters
parameters = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# Set up the plot for each parameter
plt.figure(figsize=(15, 10))

for i, param in enumerate(parameters):
    # Calculate the mean of each parameter grouped by quality
    mean_values = df.groupby('quality')[param].mean().reset_index()

    plt.subplot(4, 3, i + 1)  # Create a grid of subplots (4 rows, 3 columns)
    sns.lineplot(data=mean_values, x='quality', y=param, marker='o')
    plt.title(f'{param} by Quality')
    plt.xlabel('Quality')
    plt.ylabel(param)

    # Annotate the mean values above each point
    for j in range(len(mean_values)):
        plt.text(mean_values['quality'][j], mean_values[param][j],
                 f'{mean_values[param][j]:.2f}', ha='center', va='bottom')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()


#========= Grouped Bar Plot =========#
# Melt the DataFrame to have a long-form DataFrame
melted_df = df.melt(id_vars='quality', var_name='parameter', value_name='value')

# Group by quality and parameter, then calculate the mean
mean_values = melted_df.groupby(['quality', 'parameter'])['value'].mean().unstack()

# Drop the 'total sulfur dioxide' column
mean_values = mean_values.drop(columns=['total sulfur dioxide'])

# Plot the data
mean_values.plot(kind='bar', figsize=(15, 10))
plt.title('Average Parameter Values by Quality (Excluding Total Sulfur Dioxide)')
plt.xlabel('Quality')
plt.ylabel('Average Value')
plt.legend(title='Parameters')
plt.show()


#========= Lined Graph Plot Part One =========#
# Group by quality and parameter, then calculate the mean
mean_values = melted_df.groupby(['quality', 'parameter'])['value'].mean().unstack()

# Drop the specified columns
mean_values = mean_values.drop(columns=['total sulfur dioxide',
                                        'alcohol',
                                        'fixed acidity',
                                        'free sulfur dioxide'])

# Create a line plot with points
plt.figure(figsize=(15, 10))

for parameter in mean_values.columns:
    plt.plot(mean_values.index, mean_values[parameter], marker='o', label=parameter)  # 'o' marker for points

    # Add numerical values above each node
    for i, value in enumerate(mean_values[parameter]):
        plt.text(mean_values.index[i], value, f'{value:.2f}', ha='center', va='bottom')  # Format to 2 decimal places

plt.title('Average Parameter Values by Quality Part 2: \n(Excluding chlorides, citric acid, density,'
          ' pH, sulphates, residual sugar, volatile acidity)')
plt.xlabel('Quality')
plt.ylabel('Average Value')
plt.legend(title='Parameters')
plt.xticks(mean_values.index)  # Ensure x-ticks are set to quality values
plt.grid(True)
plt.show()


#========= Lined Graph Plot Part Two =========#
# Group by quality and parameter, then calculate the mean
mean_values = melted_df.groupby(['quality', 'parameter'])['value'].mean().unstack()

# Drop the specified columns
mean_values = mean_values.drop(columns=['chlorides',
                                        'citric acid',
                                        'residual sugar',
                                        'density',
                                        'pH',
                                        'sulphates',
                                        'volatile acidity'])

# Create a line plot with points
plt.figure(figsize=(15, 10))

for parameter in mean_values.columns:
    plt.plot(mean_values.index, mean_values[parameter], marker='o', label=parameter)  # 'o' marker for points

    # Add numerical values above each node
    for i, value in enumerate(mean_values[parameter]):
        plt.text(mean_values.index[i], value, f'{value:.2f}', ha='center', va='bottom')  # Format to 2 decimal places

plt.title('Average Parameter Values by Quality Part 1: \n(Excluding Total Sulfur Dioxide, Free Sulfur Dioxide, Alcohol)')
plt.xlabel('Quality')
plt.ylabel('Average Value')
plt.legend(title='Parameters')
plt.xticks(mean_values.index)  # Ensure x-ticks are set to quality values
plt.grid(True)
plt.show()

after = df

# Specify number of rows and columns before changes
rows, columns = before.shape
print("\nBefore any changes: ")
print(f"Number of rows: {rows}, Number of columns: {columns}")

# Specify number of rows and columns after changes
rows, columns = after.shape
print("\nAfter any changes: ")
print(f"Number of rows: {rows}, Number of columns: {columns}")

# Save processed dataframe into csv file.
df.to_csv('processed_wine_data.csv', index=False)


#========================================================#
    #========== AI Training and Building ==========#
#========================================================#
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into input features (X) and target variable (y)
X = df.drop(columns=["quality"])  # Features (the columns without 'quality')
y = df["quality"]  # Target variable (quality score)

# Use StandardScaler to scale the input features to have a mean of 0 and std of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler and transform the features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15)

# Build the Neural Network Model
model = tf.keras.models.Sequential()

# Input layer and hidden layers
model.add(tf.keras.layers.Dense(256, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Second hidden layer
model.add(tf.keras.layers.Dense(64, activation='relu'))  # Third hidden layer

# Output layer
model.add(tf.keras.layers.Dense(1, activation='linear'))  # Linear activation for regression output

# Compile and train model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mape'])  # Use MSE for loss and MAPE for evaluation
model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 9: Evaluate the Model on the Test Data
loss, mape = model.evaluate(X_test, y_test)

# Print the final results
print(f"Test MAE (Mean Absolute Error): {loss}")  # MSE is calculated as part of the evaluation
print(f"Test MAPE (Mean Absolute Percentage Error): {mape}")


#======= Showing some examples of predictions =======#
#       [fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur, total sulfur, density, pH, sulphates, alcohol]
wine1 = [9.0          , 0.660           , 0.17       , 3.0           , 0.077    , 5.0        , 13.0        , 0.99760, 3.29, 0.55   , 10.40] #quality = 5
wine2 = [11.6         , 0.230           , 0.57       , 1.80          , 0.074    , 3.0        , 8.0         , 0.99810, 3.14, 0.70   , 9.90]  #quality = 6
wine3 = [10.5         , 0.510           , 0.64       , 2.40          , 0.107    , 6.0        , 15.0        , 0.99730, 3.09, 0.66   , 11.80] #quality = 7
wine4 = [7.4          , 0.360           , 0.30       , 1.8           , 0.074    , 17.0       , 24.0        , 0.99419, 3.24, 0.70   , 11.40] #quality = 8

# Reshape the input data into a 2D array (1 row, 11 columns)
wine1_reshaped = np.array(wine1).reshape(1, -1)
wine2_reshaped = np.array(wine2).reshape(1, -1)
wine3_reshaped = np.array(wine3).reshape(1, -1)
wine4_reshaped = np.array(wine4).reshape(1, -1)

# Preprocess the new data (scale it using the same scaler)
wine1_scaled = scaler.transform(wine1_reshaped)
wine2_scaled = scaler.transform(wine2_reshaped)
wine3_scaled = scaler.transform(wine3_reshaped)
wine4_scaled = scaler.transform(wine4_reshaped)

# Use the trained model to make a prediction
prediction1 = model.predict(wine1_scaled)
prediction2 = model.predict(wine2_scaled)
prediction3 = model.predict(wine3_scaled)
prediction4 = model.predict(wine4_scaled)

# Display the prediction
print(f"Predicted Wine 1 Quality: {prediction1[0][0]}, Actual quality value = 5")  # Prediction is an array, we want the scalar value
print(f"Predicted Wine 2 Quality: {prediction2[0][0]}, Actual quality value = 6")
print(f"Predicted Wine 3 Quality: {prediction3[0][0]}, Actual quality value = 7")
print(f"Predicted Wine 4 Quality: {prediction4[0][0]}, Actual quality value = 8")
