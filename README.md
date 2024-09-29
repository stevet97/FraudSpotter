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

### Data Preparation
Customer data, such as demographic information and financial history, is loaded and explored to understand its structure.

#### Loading and Exploring the Dataset
```python
file_path = '/content/drive/My Drive/SOM/Credit_Card_Applications.csv'
dataset = load_dataset(file_path)

# Displaying the first few rows of the dataset to understand its structure
print(dataset.head())
```

### Data Preprocessing
After loading the data, it needs to be prepared for training. This involves splitting the dataset into features and labels, and scaling the features to a suitable range.

#### Splitting and Scaling the Data
```python
def preprocess_data(dataset):
    x = dataset.iloc[:, :-1].values  # Use all rows, all columns except the last one as features
    y = dataset.iloc[:, -1].values   # Use all rows, the last column as labels

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    x = sc.fit_transform(x)

    return x, y, sc

# Preprocess the data
x, y, sc = preprocess_data(dataset)
print(f"Number of features: {x.shape[1]}")
```

### SOM Training
Train a Self-Organizing Map (SOM) to detect anomalies. The `MiniSom` library is used for this purpose.

#### Training the SOM
```python
def train_som(data, x_dim=8, y_dim=8, input_len=15, sigma=1.0, learning_rate=0.5, num_iterations=100):
    som = MiniSom(x=x_dim, y=y_dim, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data)
    som.train_random(data=data, num_iteration=num_iterations)
    return som

# Train SOM
som = train_som(x, input_len=x.shape[1])
print("SOM Trained Successfully.")
```

### Fraud Detection
Once the SOM is trained, use it to identify potential anomalies in the dataset by mapping input data to neurons and identifying outliers.

#### Identifying Potential Frauds
```python
def identify_frauds(som, data):
    mappings = som.win_map(data)
    
    # Example: Hard-coded neuron locations representing outliers (this logic could be adapted)
    frauds_list = []

    # Add neurons to the list of frauds if they contain data points
    for coordinates in [(6, 2), (1, 1)]:
        if coordinates in mappings and len(mappings[coordinates]) > 0:
            frauds_list.extend(mappings[coordinates])

    frauds = np.array(frauds_list)
    return frauds

# Find fraudulent data points
frauds = identify_frauds(som, x)
```

### Visualization
Visualizing the results helps in better understanding the anomalies. A distance map is plotted to highlight areas with high discrepancy, which represent potential frauds.

#### Visualizing the SOM
```python
def plot_som(som):
    plt.figure(figsize=(10, 10))
    plt.title("SOM Distance Map")
    plt.xlabel("SOM Grid X-axis")
    plt.ylabel("SOM Grid Y-axis")
    plt.pcolor(som.distance_map().T)
    plt.colorbar(label='Distance')
    plt.show()

# Plot SOM's distance map
plot_som(som)
```
The distance map plot shows clusters and the anomalies in different colors. Areas with higher values (brighter colors) indicate possible outliers.

### Exporting Results
After detecting potential frauds, the data points identified as suspicious are exported for further analysis.

#### Exporting Fraudulent Data
```python
def save_frauds_to_csv(frauds_data, output_path='fraudulent_records.csv'):
    if len(frauds_data) > 0:
        df_frauds = pd.DataFrame(frauds_data)
        df_frauds.to_csv(output_path, index=False)
        print(f"Fraudulent data points saved to {output_path}")
    else:
        print("No fraudulent data points found to save.")

save_frauds_to_csv(frauds)
```

## Use Cases
- **Credit Card Fraud Detection**: Identify suspicious credit card applications.
- **Insurance Fraud Detection**: Detect fraudulent claims in insurance data.
- **Healthcare Anomaly Detection**: Identify unusual patient records.

## Installation & Usage
### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/FraudSpotter.git
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage
Run the Jupyter Notebook to understand and execute the process:
```sh
jupyter notebook FraudSpotter.ipynb
```

## Contributing
Contributions are welcome! Feel free to open issues or pull requests. Any contribution that improves functionality, documentation, or extends the scope of the project is highly appreciated.

## Models Used
- **Self-Organizing Maps (SOM)**: Used for unsupervised anomaly detection.

## Interesting Tidbit: Self-Organizing Maps
Self-Organizing Maps are a type of artificial neural network that uses unsupervised learning to produce a low-dimensional representation of high-dimensional data. This property makes them particularly useful for visualizing clusters and detecting anomalies, as they provide intuitive insights into the data distribution.

## Connect with Me
- **LinkedIn**: (https://www.linkedin.com/in/stephen-thomas-6037931b6/)
- **Email**: (mailto:stephenthomas382@gmail.com)
