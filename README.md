### Tutorial: Python's Significance in Materials Science

#### Introduction to Python in Materials Science

**Overview:**
Python is highly valued in materials science due to its simplicity and powerful libraries designed for scientific computations and data analysis.

**Why Python in Materials Science:**
- **Versatility:** Python offers adaptable libraries like NumPy, Pandas, SciPy, and Matplotlib, ideal for diverse materials science applications.
- **Ease of Use:** Its simple syntax makes it easy to learn and use, allowing researchers to focus on problem-solving.
- **Rich Ecosystem:** Python’s libraries facilitate efficient data handling, numerical computations, and visualization, essential in materials science.

**Significance in Materials Science:**
- **Data Handling:** Python efficiently manages and manipulates large datasets prevalent in materials science.
- **Numerical Computing:** Libraries like NumPy enable optimized numerical operations, crucial for simulations and analysis.
- **Visualization:** Tools like Matplotlib aid in visualizing complex data, aiding in understanding research findings.

#### Python's Popularity in Scientific Computing

**Overview:**
Python is a preferred choice for scientific computing due to its robustness, extensive libraries, and a supportive developer community.

**Key Reasons for Python's Popularity:**
- **Community Support:** Python’s large community continually develops scientific packages and
- **Interoperability:** Python integrates well with other languages, facilitating access to established codes.
- **Accessibility:** Being open-source, Python offers a wide array of tools without cost barriers.

**Applications in Scientific Computing:**
- **Simulation and Modeling:** Python supports computational methods vital for materials modeling and simulations.
- **Data Analysis:** Its capabilities enable efficient processing and analysis of experimental and computational data.
- **Machine Learning:** Python’s machine learning libraries aid in predictive models for materials properties.

---

# Environments
When working with Python, you have various environments for execution:

1. **Windows Powershell:** If you're using a Windows system and are familiar with Powershell, it's a powerful command-line tool that can execute Python scripts. [CLICK HERE FOR DIRECTIONS](https://www.example.com)

2. **Mac Terminal:** For Mac users comfortable with BASH, the Terminal offers an environment to run Python code. [CLICK HERE FOR DIRECTIONS](https://www.example.com)

3. **Linux:** Similar to Mac, Linux users can leverage their system's Terminal to execute Python scripts.  [CLICK HERE FOR DIRECTIONS](https://www.example.com)

4. **Chapel Hills Longleaf/Dogwood Cluster:** Accessing Chapel Hills Longleaf or Dogwood clusters allows for high-performance computing (HPC), but access may require a three-week process for approval and setup.  [CLICK HERE FOR DIRECTIONS](https://www.example.com)

5. **Google Colab:** Utilizing Google Colab, a cloud-based platform, requires a Google account. Colab provides a collaborative Jupyter notebook environment for executing Python code directly in a browser. [Access Google Colab here](https://colab.research.google.com/).

Each environment offers distinct advantages and is suited to different purposes. Throughout this tutorial series, we'll explore these environments in detail, understanding their functionalities and learning how to execute Python code effectively in each of them.

For the first half of this workshop we will use google colab, if you are planing on needing to complete a research project with a large data set I recomend requesting access to Chapel Hills HPC Resources.  


## Using Jupyter Notebook on Google Colab: A Tutorial
This tutorial provides an overview of using Jupyter Notebooks on Google Colab. Explore the functionalities for coding and data analysis!

### Step 1: Access Google Colab
1. Open your web browser and go to [Google Colab](https://colab.research.google.com/).
2. Sign in to your Google account or create one.
3. Once signed in, you'll be directed to the Colab homepage.

### Step 2: Create a New Notebook
1. Click on the "New Notebook" button on the top left or go to `File > New Notebook`.
2. A new Jupyter Notebook will open in a new tab with the filename `UntitledX.ipynb`.

### Step 3: Understanding the Interface
1. **Toolbar:** Various options for saving, adding cells, running cells, etc.
2. **Code Cells:** Write and execute Python code here.
3. **Text Cells:** Add explanations in Markdown format.
4. **Runtime Menu:** Manage the notebook's runtime environment.

### Step 4: Writing and Executing Code
1. Write Python code in a code cell.
2. Execute code cells with `Shift + Enter` or the Play button.
3. Output appears below the cell.

### Step 5: Adding Text Explanations (Markdown Cells)
1. Use the `+ Text` button or `Insert > Text cell` to add text.
2. Format text using Markdown syntax.

### Step 6: Saving and Sharing Notebooks
1. Save with `File > Save` or `Ctrl + S`.
2. Share via `Share` in the upper right corner.

### Step 7: Managing Runtime
1. Change runtime type or Python version in `Runtime > Change runtime type`.
2. Manage sessions or reset runtimes.

### Step 8: Closing and Exiting
1. To close a notebook, go to `File > Close Notebook`.
2. Exit Google Colab by closing the browser tab.

---

# Python Syntax Basics Tutorial

## Variables and Assignment

In Python, variables are containers used to store data values. Here's how you assign values to variables:

```python
# Assigning values to variables
x = 5  # integer
y = 3.14  # float
name = "Alice"  # string
is_valid = True  # boolean
```

## Data Types

Python supports various data types:

- **Integers:** Whole numbers without decimals.
- **Floats:** Numbers with decimals or in exponential form.
- **Strings:** Sequence of characters enclosed in single/double quotes.
- **Booleans:** Represents True or False values.

```python
# Examples of different data types
integer_value = 10
float_value = 3.14
string_value = "Hello, Python!"
boolean_value = True
```

## Loops and Conditional Statements

### Loops

**For Loops:** Iterate over a sequence (like a list, tuple, string, etc.).

```python
# Example of a for loop
for i in range(5):  # Loop through numbers 0 to 4
    print(i)  # Print each number
```

**While Loops:** Execute a block of code as long as the condition is True.

```python
# Example of a while loop
counter = 0
while counter < 5:
    print(counter)
    counter += 1
```

### Conditional Statements

**If-Else Statements:** Execute different blocks of code based on conditions.

```python
# Example of if-else statement
x = 10
if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")
```

# Functions and Libraries in Python

## Functions

### Introduction to Functions

In Python, a function is a block of organized, reusable code that performs a specific task. It allows you to divide your code into manageable parts and promote reusability. Here we cover the basics of functions in Python, including defining functions, using built-in functions, an introduction to libraries, and how to use popular libraries such as `math`, `numpy`, and `pandas`.

### Defining a Function

You can define a function using the `def` keyword followed by the function name and parameters within parentheses.

```python
# Defining a simple function
def greet(name):
    print(f"Hello, {name}!")

# Calling the function
greet("Alice")
```

### Built-in Functions

Python comes with a variety of built-in functions that are readily available for use.

```python
# Using built-in functions
number_list = [3, 7, 1, 5, 9]

# Finding the length of a list
print(len(number_list))

# Finding the maximum value in a list
print(max(number_list))

# Sorting a list
sorted_list = sorted(number_list)
print(sorted_list)
```

## Libraries

### Introduction to Libraries

Libraries in Python contain pre-written code/modules that provide functionalities to perform specific tasks.

### Using Libraries

To use a library, you need to import it into your script using the `import` statement.

```python
# Importing the math library
import math

# Calculating the square root
result = math.sqrt(25)
print(result)

# Using the constant pi
print(math.pi)
``` 

### Popular Libraries

Python has numerous powerful libraries for various purposes such as NumPy for numerical computations, Pandas for data manipulation, Matplotlib for data visualization, and more. Functions are essential blocks of code for organizing tasks, while libraries provide a wide range of functionalities to accomplish diverse programming tasks in Python. Mastering functions and utilizing libraries can significantly enhance your coding efficiency and capabilities.


```python
# Using NumPy for arrays
import numpy as np

# Creating a NumPy array
my_array = np.array([1, 2, 3, 4, 5])
print(my_array)

# Using Pandas for data manipulation
import pandas as pd

# Creating a Pandas DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df)
``` 


## Conclusion

Note, this is a foundational understanding of Python's syntax basics, including variables, data types, loops (for and while), conditional statements (if-else) and Functions. With these fundamental concepts, you can start writing Python code and explore more advanced functionalities within the language.

---
# Introduction to NumPy

## Role of NumPy
NumPy, short for Numerical Python, is a fundamental package in Python used for numerical computing. It's particularly helpful for handling arrays and performing various mathematical operations on them. NumPy provides a high-performance multidimensional array object (`numpy.ndarray`) along with tools for working with these arrays. Here I introduce the role of NumPy in handling arrays, creating arrays, exploring array attributes, performing mathematical operations, and using aggregation functions for numerical computations.

## Handling Arrays

### Creating Arrays

NumPy arrays can be created using lists or other array-like sequences.

```python
import numpy as np

# Creating a NumPy array from a list
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
print(my_array)
```

### Array Attributes

NumPy arrays have attributes such as shape, size, data type, etc., that provide information about the array.

```python
# Array attributes
print(my_array.shape)  # Shape of the array
print(my_array.size)  # Number of elements in the array
print(my_array.dtype)  # Data type of the array elements
```

## Numerical Computations

### Mathematical Operations

NumPy allows performing mathematical operations on arrays element-wise.

```python
# Mathematical operations
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Element-wise addition
result = array1 + array2
print(result)

# Element-wise multiplication
result = array1 * array2
print(result)
```

### Aggregation Functions

NumPy provides functions for aggregation over arrays like sum, mean, min, max, etc.

```python
# Aggregation functions
numbers = np.array([1, 2, 3, 4, 5])

# Sum of array elements
print(np.sum(numbers))

# Mean of array elements
print(np.mean(numbers))

# Maximum value in the array
print(np.max(numbers))
``` 

## Conclusion

NumPy is a powerful library for numerical computations and array manipulations in Python. Its ability to handle arrays efficiently makes it a cornerstone for scientific computing, data analysis, and machine learning tasks. 

Note: please check out the documentation for NumPy for more tutorials and functionalities [CLICK HERE FOR NumPy DOCUMENTATION](https://numpy.org/doc/stable/)
```


First assignment  [CLICK HERE FOR DIRECTIONS](https://www.example.com)
