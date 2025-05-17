import json

# Function to correct the JSON structure
def correct_json( output_file):
    with open('dataset labeled/dev.json', 'r') as f:
        data = f.read()

    # Remove newline characters from each line and add ',' at the end (except for the last line)
    corrected_lines = [line.strip() + (',' if line.strip()[-1] == '}' else '') for line in data[:-1]]
    # Add last line without a comma at the end
    corrected_lines.append(data[-1].strip())

    # Join the lines with ',' and enclose them in '[' and ']'
    corrected_data = '[' + ''.join(corrected_lines) + ']'
    
    print(corrected_data)
    # Load the corrected data as JSON
    json_data = json.loads(corrected_data)

    

    # Write the corrected data to a new file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

# Example usage
input_file = 'dataset labeled/dev.json'
output_file = 'corrected_dev.json'
correct_json( output_file)
