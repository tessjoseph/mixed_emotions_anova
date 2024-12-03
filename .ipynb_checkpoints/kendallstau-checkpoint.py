import pandas as pd
import glob
from scipy.stats import kendalltau

# Path to the directory containing the CSV files
path = "results/final/gpt4/*"
 
# List of all CSV files
files = glob.glob(path)

# Read and concatenate all CSV files into a single DataFrame
df_list_4 = [pd.read_csv(file) for file in files]
combined_df_gpt4 = pd.concat(df_list_4, ignore_index=True)

# Path to the directory containing the CSV files
path = "results/final/gpt3.5/*"

# List of all CSV files
files = glob.glob(path)

# Read and concatenate all CSV files into a single DataFrame
df_list_3 = [pd.read_csv(file) for file in files]
combined_df_gpt3 = pd.concat(df_list_3, ignore_index=True)

# Path to the directory containing the CSV files
path = "results/final/mistral/*"

# List of all CSV files
files = glob.glob(path)

# Read and concatenate all CSV files into a single DataFrame
df_list_m = [pd.read_csv(file) for file in files]
combined_df_mistral = pd.concat(df_list_m, ignore_index=True)

# Compute Kendall's tau for 'A' and 'B'
tau, p_value = kendalltau(combined_df_gpt3['heart_rate'], combined_df_mistral['heart_rate'])

print(f"Kendall's tau: {tau}")
print(f"P-value: {p_value}")
#retrieve all the responses of the specific diet
#see the relationship across different diets
#mul