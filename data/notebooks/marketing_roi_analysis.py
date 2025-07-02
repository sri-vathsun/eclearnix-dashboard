import pandas as pd

# Example: Load your data into df (replace with your actual data source)
df = pd.read_csv('ECLEARNIX.csv')

# Assuming Platform_Source is marketing channel
roi_df = df.groupby('Platform_Source').agg({
    'User_ID': 'count',
    'Course_Completed': 'sum'
}).rename(columns={'User_ID': 'Total_Users', 'Course_Completed': 'Conversions'}).reset_index()

roi_df['Conversion_Rate'] = roi_df['Conversions'] / roi_df['Total_Users']
roi_df.sort_values('Conversion_Rate', ascending=False, inplace=True)

print(roi_df)
