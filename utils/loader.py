import json
import yaml
import os

def load_call_data(file_path):
    """Load and validate JSON call data"""
    with open(file_path, 'r') as f:
        utterances = json.load(f)
        
        # Validate structure
        for utterance in utterances:
            if not all(key in utterance for key in ("speaker", "text", "stime", "etime")):
                raise ValueError("Invalid JSON structure")
                
        return utterances

def parse_file(uploaded_file):
    """
    Parse uploaded file (YAML or JSON)
    Returns list of utterances with speaker, text, stime, etime
    """
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Handle based on file extension
    if file_extension == 'json':
        # For JSON files, read and parse directly
        content = uploaded_file.read()
        utterances = json.loads(content)
        
        # Validate structure
        for utterance in utterances:
            if not all(key in utterance for key in ("speaker", "text", "stime", "etime")):
                raise ValueError("Invalid JSON structure")
                
        return utterances
        
    elif file_extension in ['yaml', 'yml']:
        # For YAML files, parse with PyYAML
        content = uploaded_file.read()
        data = yaml.safe_load(content)
        
        # Handle expected YAML structure with "utterances" as key
        if "utterances" in data:
            utterances = data["utterances"]
            
            # Validate structure
            for utterance in utterances:
                if not all(key in utterance for key in ("speaker", "text", "stime", "etime")):
                    raise ValueError("Invalid YAML structure")
                    
            return utterances
        else:
            raise ValueError("Invalid YAML structure: 'utterances' key not found")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def save_as_yaml(json_data, output_dir="data/sample_calls"):
    """
    Save JSON data as YAML file in the specified directory.
    Uses the JSON filename as the YAML filename.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename without extension
    filename = os.path.basename(json_data["filename"]).split('.')[0]
    output_path = os.path.join(output_dir, f"{filename}.yaml")
    
    # Format data for YAML
    yaml_data = {
        "utterances": json_data["utterances"]
    }
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    return output_path