import pandas as pd
data = pd.read_csv("Healthcare.csv")
"""
print(data.columns) tells columns in actual csv file
print(data.shape)   telss shape of it
print(data.info())  gives info like data types and non null values
print(data.head())   just shows top 5 rows of the data
print(data['Symptoms'].head()) shows the symptoms column of top 5 rows
print(data.isnull().sum()) shows the number of null values in each column"""


data['Symptoms'] = data['Symptoms'].str.split(',') # splits the symptoms column into a list of symptoms for each row, using comma as the separator
#print(data['Symptoms'].head())
data['Symptoms'] = data['Symptoms'].apply(lambda x: [i.strip() for i in x])  # removes any leading or trailing whitespace from each symptom in the list, using a lambda function and the strip() method
#print(data['Symptoms'].head())

all_symptoms = set()

for symptoms in data['Symptoms']:
    for symptom in symptoms:
        all_symptoms.add(symptom) # adds each unique symptom to the all_symptoms set, which will contain all the distinct symptoms present in the dataset

#print(len(all_symptoms))

for symptom in all_symptoms:
    data[symptom] = data['Symptoms'].apply(lambda x: 1 if symptom in x else 0)  # creates a new column for each unique symptom, and fills it with 1 if the symptom is present in the Symptoms list for that row, and 0 otherwise, using a lambda function and the apply() method

#print(data.head())

# print(len(data.columns)) # prints the number of columns in the modified dataset, which includes the original columns and the new binary columns for each unique symptom.

# print(data.columns) # prints the names of all the columns in the modified dataset, which includes the original columns and the new binary columns for each unique symptom.

data = data.drop(['Patient_ID', 'Symptoms', 'Symptom_Count'], axis=1) # drops the original columns that are no longer needed, such as Patient_ID, Symptoms, and Symptom_Count, using the drop() method and specifying the axis=1 parameter to indicate that we want to drop columns rather than rows
data = data.drop(['Gender'], axis=1) # drops the Gender column, which may not be relevant for the analysis or modeling task at hand, using the drop() method and specifying the axis=1 parameter to indicate that we want to drop columns rather than rows

#print(data.head())
#print(data.columns)

#print(data['Disease'].unique())
selected_diseases = [
    'Allergy',
    'Bronchitis',
    'Asthma',
    'Food Poisoning',
    'Sinusitis',
    'Gastritis',
    'Migraine',
    'Common Cold',
    'Ulcer',
    'Influenza'
]

data = data[data['Disease'].isin(selected_diseases)] # filters the dataset to include only rows where the Disease column contains one of the selected diseases, using the isin() method to check for membership in the selected_diseases list

#print(data['Disease'].unique()) # prints the unique values in the Disease column of the filtered dataset, which should now only include the selected diseases

#print(data.shape) # prints the shape of the filtered dataset, which should reflect the number of rows and columns after filtering for the selected diseases

data = data.reset_index(drop=True) # resets the index of the filtered dataset, dropping the old index and creating a new one that starts from 0, using the reset_index() method with the drop=True parameter to avoid adding the old index as a new column in the dataset

data.to_csv("cleaned_dataset.csv", index=False) # saves the cleaned and filtered dataset to a new CSV file named "cleaned_dataset.csv", using the to_csv() method with the index=False parameter to avoid including the index in the output file