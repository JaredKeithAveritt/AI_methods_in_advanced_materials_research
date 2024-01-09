### Homework Assignment 1: Exploring NumPy [SOLUTIONS]

#### Task 1: Basic Operations
These solutions demonstrate creating a 3x3 matrix with values 1 to 9 and replacing odd numbers in the matrix with -1 using NumPy operations.

1. **Array Creation:** Create a 3x3 matrix with values ranging from 1 to 9.

```python
import numpy as np

# Create a 3x3 matrix with values ranging from 1 to 9
matrix = np.arange(1, 10).reshape(3, 3)
print("Original Matrix:")
print(matrix)
```

Output:
```
Original Matrix:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

2. **Element Manipulation:** Replace all odd numbers in the matrix with -1.

```python
# Replace odd numbers in the matrix with -1
matrix[matrix % 2 != 0] = -1
print("\nMatrix after replacing odd numbers with -1:")
print(matrix)
```

Output:
```
Matrix after replacing odd numbers with -1:
[[-1  2 -1]
 [ 4 -1  6]
 [-1  8 -1]]
```
---

### Task 2: Statistical Analysis
These solutions demonstrate the use of NumPy functions to compute the mean, standard deviation, and correlation coefficient of randomly generated arrays.

1. **Mean and Standard Deviation:** Generate a random 1-D array of 50 elements. Compute the mean and standard deviation of this array.

```python
import numpy as np

# Generate a random 1-D array of 50 elements
random_array = np.random.rand(50)

# Compute the mean and standard deviation
mean_value = np.mean(random_array)
std_deviation = np.std(random_array)

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_deviation}")
```

2. **Correlation Coefficient:** Create two random 1-D arrays with 30 elements each and find the correlation coefficient between them.

```python
import numpy as np

# Create two random 1-D arrays with 30 elements each
array_1 = np.random.rand(30)
array_2 = np.random.rand(30)

# Find the correlation coefficient between the arrays
correlation_coefficient = np.corrcoef(array_1, array_2)[0, 1]

print(f"Correlation Coefficient: {correlation_coefficient}")
```

---

### Task 3: Array Manipulation
These solutions demonstrate how to reshape a 1-D array into a 2-D matrix and how to extract the middle row and column from the resulting matrix using NumPy's indexing and slicing capabilities.

#### 1. Reshaping Arrays

1. **Reshaping Arrays:** Create a 2-D array with 20 elements and reshape it into a 4x5 matrix.

```python
import numpy as np

# Create a 1-D array with 20 elements
arr = np.arange(20)

# Reshape the array into a 4x5 matrix
reshaped_arr = arr.reshape(4, 5)

print("Original 1-D Array:")
print(arr)
print("\nReshaped 2-D Array (4x5 matrix):")
print(reshaped_arr)
```

2. **Slicing and Indexing:** Extract the middle row and middle column from the 4x5 matrix.

**Extract the middle row and middle column from the 4x5 matrix:**

```python
import numpy as np

# Assuming 'reshaped_arr' is the 4x5 matrix from the previous task

# Extracting the middle row (row index 2)
middle_row = reshaped_arr[2, :]
print("\nMiddle Row:")
print(middle_row)

# Extracting the middle column (column index 2)
middle_column = reshaped_arr[:, 2]
print("\nMiddle Column:")
print(middle_column)
```

---

#### Task 4: Universal Functions

These solutions utilize NumPy functions (`np.sqrt()` for square root and `np.sin()`, `np.cos()`, `np.tan()` for trigonometric calculations) to perform the required operations on arrays created or provided. Students should get results similar to these upon executing the code.

1. **Square Root:** Generate an array of 10 random integers and find the square root of each element.

```python
import numpy as np

# Generate an array of 10 random integers
arr = np.random.randint(1, 100, 10)

# Calculate square root of each element
sqrt_arr = np.sqrt(arr)

print("Original Array:", arr)
print("Square Roots:", sqrt_arr)
```

2. **Trigonometric Functions:** Create an array of angles in radians and compute the sine, cosine, and tangent of each angle.

```python
import numpy as np

# Create an array of angles in radians
angles = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

# Compute sine, cosine, and tangent of each angle
sin_vals = np.sin(angles)
cos_vals = np.cos(angles)
tan_vals = np.tan(angles)

print("Angles (radians):", angles)
print("Sine Values:", sin_vals)
print("Cosine Values:", cos_vals)
print("Tangent Values:", tan_vals)
```

---

#### Task 4: Universal Functions
This solution's code uses Python with NumPy to compute the areas of circles based on random radii. It defines two functions: `calculate_area()` and `calculate_area_with_numpy()`. The latter function operates on a NumPy array containing 10 randomly generated radii to calculate their respective areas. The resulting areas are then displayed for each radius in the array.

```python
import numpy as np

# Function to calculate area without NumPy
def calculate_area(radius):
    return np.pi * (radius ** 2)

# Function to calculate area with NumPy array of 10 random radii
def calculate_area_with_numpy(radius_array):
    return np.pi * (radius_array ** 2)

# Creating an array of 10 random radii
random_radii = np.random.uniform(1, 10, 10)

# Calculating areas using the function with NumPy
areas = calculate_area_with_numpy(random_radii)

# Displaying the calculated areas
for i, area in enumerate(areas):
    print(f"Area {i+1}: {area:.2f}")
```

--- 

### Task 6: Additional Challenge (Optional)

Design a scenario where a student needs to perform an operation or analysis using NumPy and provide step-by-step instructions on how they would utilize NumPy to solve the problem. Encourage creativity and practical application.

#### Solution Example:

This Python code snippet simulates and visualizes the trajectory of a projectile motion. It uses NumPy for calculations such as position over time and employs Matplotlib for visualization.

**Scenario:** Calculate and visualize the trajectory of a projectile motion using NumPy.

1. **Step-by-Step Instructions:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
initial_velocity = 20  # m/s
launch_angle = np.deg2rad(30)  # Convert degrees to radians
acceleration_due_to_gravity = 9.81  # m/s^2

# Time values
time_values = np.linspace(0, 4, 100)  # 0 to 4 seconds, 100 data points

# Calculate horizontal and vertical components of velocity
horizontal_velocity = initial_velocity * np.cos(launch_angle)
vertical_velocity = initial_velocity * np.sin(launch_angle)

# Calculate horizontal and vertical positions
horizontal_position = horizontal_velocity * time_values
vertical_position = (vertical_velocity * time_values) - (0.5 * acceleration_due_to_gravity * (time_values ** 2))

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(horizontal_position, vertical_position)
plt.title('Projectile Motion')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.grid(True)
plt.show()
```

The Python code bellow produces a trajectory plot depicting the simulated projectile motion. The following comments elucidate the physics underlying the projectile's movement and explain the utilization of NumPy for the calculations. This response exemplifies the utilization of NumPy to simulate and visualize the projectile's trajectory while succinctly explaining the relevant physics principles.

**Example Response:**

The provided Python code simulates the projectile motion of an object launched at an initial velocity of 20 m/s at an angle of 30 degrees with respect to the horizontal axis. This is achieved by utilizing NumPy for handling numerical computations, particularly for calculating horizontal and vertical components of velocity, as well as the positions over a specified time range. The linspace function in NumPy creates 100 evenly spaced time points between 0 and 4 seconds. 

The calculations for horizontal and vertical positions consider the effects of gravity on the object. The resulting plot displays the trajectory of the projectile, showcasing how its position changes over time.

The physics behind this simulation involves utilizing the equations of motion for projectile motion. NumPy aids in performing numerical calculations efficiently, allowing for the determination of the object's position at different time intervals. The resulting trajectory plot visually represents the projectile's path in a 2D space, depicting its horizontal and vertical distances as time progresses.

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
initial_velocity = 20  # m/s
launch_angle = np.deg2rad(30)  # Convert degrees to radians
acceleration_due_to_gravity = 9.81  # m/s^2

# Time values
time_values = np.linspace(0, 4, 100)  # 0 to 4 seconds, 100 data points

# Calculate horizontal and vertical components of velocity
horizontal_velocity = initial_velocity * np.cos(launch_angle)
vertical_velocity = initial_velocity * np.sin(launch_angle)

# Calculate horizontal and vertical positions
horizontal_position = horizontal_velocity * time_values
vertical_position = (vertical_velocity * time_values) - (0.5 * acceleration_due_to_gravity * (time_values ** 2))

# Plot the trajectory
plt.figure(figsize=(8, 6))
plt.plot(horizontal_position, vertical_position)
plt.title('Projectile Motion')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.grid(True)
plt.show()
```

## INSERT PLOT ## 

---




