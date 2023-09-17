import os


folder_path = '/Users/tina/Desktop/OutputImages/Spanish/7maridos'

all_files = os.listdir(folder_path)

# Filter the list to include only files with the desired extension
filtered_files = [file for file in all_files if file.endswith('xml')]

for file in filtered_files:
    with open(os.path.join(folder_path, file), "r") as f:
        xml_content = f.read()

    # Remove the first character from the content

    if xml_content[0] == "'":
        modified_xml_content = xml_content[1:]
        # Write the modified content back to the file
        with open(os.path.join(folder_path,file), "w") as file:
            file.write(modified_xml_content)





