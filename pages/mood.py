import matplotlib as mat
import pandas as pd
from pathlib import Path

# Get current file’s path
current_file = Path(__file__)

# Construct path to CSV
csv_path = current_file.parent.parent / 'databases' / 'mood.csv'

#dataframe to print
df = pd.read_csv(csv_path)

#debug
print(df.to_string())

