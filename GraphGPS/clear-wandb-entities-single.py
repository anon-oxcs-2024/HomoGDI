import os

# Define the directory containing the .yaml files
filename = '/Users/linusbao123/Documents/Oxford/Thesis/Graph ML/repo_submit/efficient attempt git/Homomorphism-GRL/GraphGPS/cleaning-test/no-entity/bad.yaml'

if filename.endswith('.yaml'):
    
    file_path = os.path.join(directory_path, filename)
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Filter out the line to be deleted
    lines = [line for line in lines if line.strip() != line_to_delete]
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

