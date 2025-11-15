"""
This defines how the criterias in the json files are read and processed.
"""

import json
from typing import NamedTuple

def extract_text_only(json_file_path: str) -> list:
    """Extract only the text content of all criteria"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dsm_data = json.load(f)
    
    criteria_texts = []
    
    for disorder in dsm_data:
        criteria_list = disorder.get("criteria", [])
        for criterion in criteria_list:
            text = criterion.get("text", "").strip()
            if text:  # Only add non-empty text
                criteria_texts.append(text)
    
    return criteria_texts

def extract_text_and_id(json_file_path: str) -> list:
    """Extract text and ID as tuples"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dsm_data = json.load(f)
    
    criteria_data = []
    
    for disorder in dsm_data:
        criteria_list = disorder.get("criteria", [])
        for criterion in criteria_list:
            criterion_id = criterion.get("id", "")
            text = criterion.get("text", "").strip()
            
            if text:
                criteria_data.append((criterion_id, text))
    
    return criteria_data

# Alternative: Dictionary format
def extract_text_and_id_dict(json_file_path: str) -> list:
    """Extract as list of dictionaries"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dsm_data = json.load(f)
    
    criteria_data = []
    
    for disorder in dsm_data:
        criteria_list = disorder.get("criteria", [])
        for criterion in criteria_list:
            if criterion.get("text", "").strip():
                criteria_data.append({
                    "id": criterion.get("id", ""),
                    "text": criterion.get("text", "").strip()
                })
    
    return criteria_data

def extract_complete_data(json_file_path: str) -> list:
    """Extract all information: diagnosis, ID, and text"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dsm_data = json.load(f)
    
    criteria_data = []
    
    for disorder in dsm_data:
        diagnosis = disorder.get("diagnosis", "")
        criteria_list = disorder.get("criteria", [])
        
        for criterion in criteria_list:
            text = criterion.get("text", "").strip()
            if text:
                criteria_data.append({
                    "diagnosis": diagnosis,
                    "criterion_id": criterion.get("id", ""),
                    "text": text
                })
    
    return criteria_data

# Alternative: Named tuples for better structure
class CriteriaEntry(NamedTuple):
    diagnosis: str
    criterion_id: str
    text: str

def extract_complete_named_tuples(json_file_path: str) -> list:
    """Extract as named tuples for type safety"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dsm_data = json.load(f)
    
    criteria_data = []
    
    for disorder in dsm_data:
        diagnosis = disorder.get("diagnosis", "")
        criteria_list = disorder.get("criteria", [])
        
        for criterion in criteria_list:
            text = criterion.get("text", "").strip()
            if text:
                entry = CriteriaEntry(
                    diagnosis=diagnosis,
                    criterion_id=criterion.get("id", ""),
                    text=text
                )
                criteria_data.append(entry)
    
    return criteria_data

# Usage
if __name__ == "__main__":
    criteria_texts = extract_text_only("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    print(f"Total criteria: {len(criteria_texts)}")
    print(f"Example: {criteria_texts[0][:100]}...")

    criteria_with_ids = extract_text_and_id("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    print(f"Example: ID='{criteria_with_ids[0][0]}', Text='{criteria_with_ids[0][1][:50]}...'")

    criteria_dicts = extract_text_and_id_dict("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    print(f"Example: ID='{criteria_dicts[0]['id']}', Text='{criteria_dicts[0]['text'][:50]}...'")

    complete_data = extract_complete_data("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    print(f"Example: {complete_data[0]}")

    complete_named_tuples = extract_complete_named_tuples("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    print(f"Example: Diagnosis='{complete_named_tuples[0].diagnosis}', Text='{complete_named_tuples[0].text[:50]}...'")

