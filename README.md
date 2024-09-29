# FraudSpotter: Detecting Anomalies in Credit Card Applications using Self-Organizing Maps

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Solved](#problem-solved)
3. [Key Features](#key-features)
4. [Technologies Used](#technologies-used)
5. [How It Works](#how-it-works)
   - [Data Preparation](#data-preparation)
   - [Data Preprocessing](#data-preprocessing)
   - [SOM Training](#som-training)
   - [Fraud Detection](#fraud-detection)
   - [Visualization](#visualization)
6. [Use Cases](#use-cases)
7. [Installation & Usage](#installation--usage)
8. [Contributing](#contributing)
9. [Models Used](#models-used)
10. [Interesting Tidbit: Self-Organizing Maps](#interesting-tidbit-self-organizing-maps)
11. [Connect with Me](#connect-with-me)

## Project Overview
FraudSpotter is a fraud detection tool that uses **Self-Organizing Maps (SOM)** to detect anomalies in credit card applications. This unsupervised approach clusters applications and highlights potential outliers which could be fraudulent. The flexibility of the SOM model allows it to be adapted across different datasets, making it practical for fraud detection use cases in various industries.

## Problem Solved
Fraud in financial institutions leads to significant losses. Traditional fraud detection methods often rely on labeled data, making it difficult to adapt to unknown patterns. FraudSpotter uses **unsupervised learning** to detect potential frauds without the need for labeled examples, providing an adaptive approach to identify new or emerging types of fraud that might go unnoticed with traditional systems.

## Key Features
- **Unsupervised Learning**: Utilizes a Self-Organizing Map to cluster data and detect anomalies without pre-labeled examples.
- **Scalable**: The model can be adapted for use with various types of datasets, offering versatility in multiple contexts.
- **Visual Insights**: Provides a clear visual representation of clusters and potential outliers for better interpretability.

## Technologies Used
- **Python 3.7+**
- **MiniSom**: For implementing Self-Organizing Maps.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Scikit-Learn**: For feature scaling.

## How It Works
**Data Preparation**

### Data Preparation
Customer data, such as demographic information and financial history, is loaded and explored to understand its structure.

Data Preprocessing
After loading the data, it needs to be prepared for training. This involves splitting the dataset into features and labels, and scaling the features to a suitable range.

Splitting and Scaling the Data
#### Loading and Exploring the Dataset
```python
file_path = '/content/drive/My Drive/SOM/Credit_Card_Applications.csv'
dataset = load_dataset(file_path)

# Displaying the first few rows of the dataset to understand its structure
print(dataset.head())
Data Preprocessing
After loading the data, it needs to be prepared for training. This involves splitting the dataset into features and labels, and scaling the features to a suitable range.

Splitting and Scaling the Data
