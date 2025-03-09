import pandas as pd

# Define the range of values and the column name
start_value = 0
end_value = 1500
column_name = 'Price_in_euros'

# Read the original CSV file
original_data = pd.read_csv('cleaned_laptop_data.csv')

# Copy the rows with values in the specified range in the specified column into a new DataFrame
new_data = original_data[(original_data[column_name] >= start_value) & (original_data[column_name] <= end_value)]

# Write the new DataFrame to a new CSV file
new_data.to_csv('new_file.csv', index=False)
