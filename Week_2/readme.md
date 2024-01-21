# Part 1: Understanding Tensors and their Usage in PyTorch

## Introduction

Lets talk about tensors and their application in PyTorch. Tensors are fundamental to deep learning, serving as the core data structure for various operations. Today, we'll explore the characteristics of tensors, how to work with them in PyTorch, and their crucial role in building neural networks.

## Table of Contents

1. [Characteristics of Tensors](#characteristics-of-tensors)
   - Rank and Shape
   - Data Types

2. [PyTorch Tensors](#pytorch-tensors)
   - Creating Tensors
   - Operations on Tensors
   - Tensor Indexing and Slicing

3. [Applications in PyTorch](#applications-in-pytorch)
   - Neural Networks
   - GPU Acceleration
   - Gradient Computation

4. [Conclusion](#conclusion)

## Characteristics of Tensors

### Rank and Shape

In PyTorch, tensors can have different ranks (number of dimensions) and shapes. Let's explore this concept with some examples:

```python
import torch

# Scalar (Rank 0)
scalar_tensor = torch.tensor(42)
print("Scalar Tensor:", scalar_tensor)

# Vector (Rank 1)
vector_tensor = torch.tensor([1, 2, 3])
print("Vector Tensor:", vector_tensor)

# Matrix (Rank 2)
matrix_tensor = torch.tensor([[1, 2], [3, 4]])
print("Matrix Tensor:\n", matrix_tensor)
```

### Data Types

The data type of a tensor determines the type of elements it can store. PyTorch supports various data types such as float, int, and double:

```python
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int)
```

## PyTorch Tensors

### Creating Tensors

PyTorch provides multiple ways to create tensors. Here are some examples:

```python
# Using torch.Tensor()
tensor_a = torch.Tensor([[1, 2], [3, 4]])

# Zeros and Ones tensors
zeros_tensor = torch.zeros((3, 3))
ones_tensor = torch.ones((2, 2))

print("Tensor A:\n", tensor_a)
print("Zeros Tensor:\n", zeros_tensor)
print("Ones Tensor:\n", ones_tensor)
```

### Operations on Tensors

Tensors support various operations like addition, subtraction, and multiplication. Let's see some examples:

```python
# Element-wise operations
tensor_b = torch.Tensor([[2, 3], [4, 5]])
result_tensor = tensor_a + tensor_b

# Matrix multiplication
matrix_mult_result = torch.matmul(tensor_a, tensor_b)

print("Result Tensor (Element-wise):\n", result_tensor)
print("Matrix Multiplication Result:\n", matrix_mult_result)
```

### Tensor Indexing and Slicing

Accessing and modifying elements in tensors can be done using indexing and slicing, similar to arrays:

```python
# Indexing
print("Element at (1, 1):", tensor_a[1, 1])

# Slicing
print("First column of Tensor A:\n", tensor_a[:, 0])
```

## Applications in PyTorch

### Neural Networks

Tensors play a crucial role in representing inputs, weights, and outputs in neural networks:

```python
# Example of a simple neural network layer
input_tensor = torch.randn((3, 3))
weights_tensor = torch.randn((3, 2))

output_tensor = torch.matmul(input_tensor, weights_tensor)
print("Neural Network Output:\n", output_tensor)
```

### GPU Acceleration

PyTorch seamlessly supports GPU acceleration for faster computations:

```python
# Move tensor to GPU
gpu_tensor = tensor_a.to('cuda')

# Perform operations on GPU
result_gpu = gpu_tensor + tensor_b.to('cuda')
```

### Gradient Computation

Tensors enable automatic differentiation in PyTorch, essential for computing gradients during optimization:

```python
# Example of gradient computation
tensor_c = torch.randn((2, 2), requires_grad=True)
loss = (tensor_c * tensor_c).sum()

# Compute gradients
loss.backward()

print("Gradient of Tensor C:\n", tensor_c.grad)
```
---

# Part 2: Introduction to Pandas for Data Manipulation and Analysis

## Introduction

Pandas is a powerful Python library widely used for data manipulation and analysis. Today, we'll explore the basics of Pandas, covering key data structures, data manipulation techniques, and common operations for working with structured data.

## Table of Contents

1. [Pandas Basics](#pandas-basics)
   - DataFrames and Series
   - Reading and Writing Data

2. [Data Manipulation with Pandas](#data-manipulation-with-pandas)
   - Indexing and Selection
   - Filtering and Sorting
   - Handling Missing Data

3. [Data Analysis with Pandas](#data-analysis-with-pandas)
   - Descriptive Statistics
   - Grouping and Aggregation
   - Merging and Joining DataFrames

4. [Conclusion](#conclusion)

## Pandas Basics

### DataFrames and Series

Pandas introduces two main data structures: **DataFrame** and **Series**.

- **DataFrame:** A two-dimensional table with rows and columns. It is similar to a spreadsheet or SQL table.

```python
import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'City': ['New York', 'San Francisco', 'Los Angeles']}

df = pd.DataFrame(data)
print(df)
```

- **Series:** A one-dimensional labeled array, similar to a column in a DataFrame.

```python
# Creating a Series
ages = pd.Series([25, 30, 22], name='Age')
print(ages)
```

### Reading and Writing Data

Pandas supports reading data from various sources, including CSV, Excel, and SQL. Here's an example of reading and writing data:

```python
# Reading data from a CSV file
df_csv = pd.read_csv('example.csv')

# Writing data to a CSV file
df.to_csv('output.csv', index=False)
```

## Data Manipulation with Pandas

### Indexing and Selection

Pandas provides versatile ways to index and select data.

```python
# Selecting a column
names = df['Name']

# Selecting multiple columns
subset = df[['Name', 'City']]

# Selecting rows based on a condition
young_people = df[df['Age'] < 30]
```

### Filtering and Sorting

Filtering data based on conditions and sorting are common operations.

```python
# Filtering based on a condition
filtered_data = df[df['City'].str.startswith('New')]

# Sorting by a column
sorted_data = df.sort_values(by='Age')
```

### Handling Missing Data

Pandas offers tools for handling missing data, such as `dropna()` and `fillna()`.

```python
# Dropping rows with missing values
df_no_missing = df.dropna()

# Filling missing values with a specific value
df_filled = df.fillna(0)
```

## Data Analysis with Pandas

### Descriptive Statistics

Pandas provides descriptive statistics for numerical data.

```python
# Descriptive statistics
summary_stats = df.describe()
```

### Grouping and Aggregation

Grouping data allows performing operations on subsets of the data.

```python
# Grouping by a column and calculating mean
mean_age_by_city = df.groupby('City')['Age'].mean()
```

### Merging and Joining DataFrames

Merging and joining help combine data from different DataFrames.

```python
# Merging two DataFrames
merged_data = pd.merge(df1, df2, on='common_column')
```

# Tutorial 1: Loading and Preprocessing Materials Data in Pandas

In this tutorial, we'll cover the basics of loading materials data into a Pandas DataFrame and preprocessing it for machine learning tasks in the field of materials science. We'll use a sample dataset and perform essential preprocessing steps to ensure the data is ready for analysis and modeling.

## Table of Contents

1. [Introduction](#introduction)
2. [Loading Materials Data](#loading-materials-data)
3. [Handling Missing Values](#handling-missing-values)
4. [Feature Engineering](#feature-engineering)
5. [Conclusion](#conclusion)

## Introduction

Materials science often involves working with complex datasets that require careful handling and preprocessing. Pandas, a powerful Python library, offers convenient tools for loading, cleaning, and transforming data. This tutorial aims to guide you through the process of loading materials data into a Pandas DataFrame and performing basic preprocessing steps.

## Loading Materials Data

In this section, we'll load a sample materials dataset into a Pandas DataFrame. For demonstration purposes, we'll assume the dataset is in a CSV file named `materials_data.csv`.

```python
import pandas as pd

# Load materials data from CSV into a Pandas DataFrame
file_path = 'materials_data.csv'
materials_df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(materials_df.head())
```

This code reads the materials data from the CSV file into a Pandas DataFrame and prints the first few rows to inspect the dataset.

## Handling Missing Values

Handling missing values is a crucial step in preprocessing materials data. Pandas provides methods to identify and deal with missing values effectively.

```python
# Check for missing values in the DataFrame
missing_values = materials_df.isnull().sum()

# Drop rows with missing values or fill missing values with appropriate strategies
materials_df_cleaned = materials_df.dropna()  # Drop rows with missing values
# Alternatively, fill missing values with the mean or median:
# materials_df_cleaned = materials_df.fillna(materials_df.mean())

# Display the cleaned DataFrame
print(materials_df_cleaned.head())
```

This code checks for missing values in the DataFrame, drops rows with missing values, or fills missing values using mean or median.

## Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve the quality of the data for machine learning models.

```python
# Assume 'composition' column contains chemical compositions (e.g., 'Fe2O3')
# Extract the element symbols and create new columns
materials_df['element_1'] = materials_df['composition'].str.extract(r'([A-Za-z]+)\d*')
materials_df['element_2'] = materials_df['composition'].str.extract(r'([A-Za-z]+)\d*', expand=False, second=True)

# Display the DataFrame with new features
print(materials_df.head())
```

This code extracts element symbols from the 'composition' column and creates new 'element_1' and 'element_2' columns.

---

%% https://training.galaxyproject.org/training-material/topics/statistics/tutorials/clustering_machinelearning/tutorial.html
