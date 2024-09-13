import os

# Define the directory containing the .yaml files
directory_path = '/Users/linusbao123/Documents/Oxford/Thesis/Graph ML/repo_submit/efficient attempt git/Homomorphism-GRL/GraphGPS/cleaning-test'

# Define the line to be deleted (exact match)
line_to_delete = '  entity: linusbao'

#recurisvely get all files in directory
cfgs=[]
for root, dirs, files in os.walk(directory_path):
    for file in files:
        cfgs.append(os.path.abspath(os.path.join(root, file)))

# Iterate over each file in the directory
for filename in cfgs:
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

        print(f'Processed file: {filename}')

print('All files processed.')
