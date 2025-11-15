import os
import sys

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach project root (/home/oscartsao/Developer/Psy_Agent)
project_root = os.path.join(current_dir, '..') 
project_root = os.path.abspath(project_root)

# Add to Python path
sys.path.insert(0, project_root)


from src.Dataloader.criteria_reader import extract_complete_named_tuples

if __name__ == "__main__":
    criteria = extract_complete_named_tuples("Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json")
    for c in criteria:
        print(c.diagnosis, c.criterion_id, c.text)