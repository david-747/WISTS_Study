import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from scipy.stats import pearsonr

import nltk

# Download necessary NLTK data (only need to run this once)
#nltk.download('punkt')      # For tokenizing text into words
#nltk.download('stopwords')  # For a list of common stopwords
#nltk.download('wordnet')    # For lemmatization


def cronbach_alpha(items_df):
    items_df = items_df.dropna(axis=0)  # Drop rows with missing values
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    n_items = items_df.shape[1]
    if total_var == 0 or n_items <= 1:
        return np.nan
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


try:
    df_all = pd.read_csv("Ressources/results-survey539458_10062025.csv")

except FileNotFoundError:
    print(f"Error: The file 'Ressources/results-survey539458_10062025.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    print(traceback.format_exc())
#only take participants that completed the survey
df_completed = df_all[df_all["lastpage"] == 10]

#remove the first seven columns id, submitdate, last page, ...
df_completed = df_completed.drop(columns=df_all.columns[:7])

#remove the remaining columns which are timings of the question groups
df_completed = df_completed.drop(columns=df_all.columns[111:])

#remove all columns which have only nulls, i.e. some "other" text fields and the LLM disclaimer
df_completed = df_completed.dropna(axis=1, how = 'all')

#print(df_completed[df_completed["DemQ02"] == 199])

#filter ages to realistic range
df_completed = df_completed[df_completed["DemQ02"] < 120]
#df_completed.loc[df_completed["DemQ02"] == 199, "DemQ02"] = 50

#test if one outlier is removed
#print(df_completed[df_completed["DemQ02"] == 199])  # Should be empty
#print(df_completed[df_completed["DemQ02"] == 50])   # Should show the updated participant


#import matplotlib.pyplot as plt

#df_completed["DemQ02"].hist(bins=20)
#plt.title("Age Distribution")
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.show()

#print(df_completed["DemQ02"].describe())

#print(df_completed[df_completed["DemQ02"] < 20])

field_map = {
    "Information Technology": [
        "it", "it-consulting", "it-governance / consulting", "it/ai",
        "ai", "data science", "software", "computer science",
        "information management", "study robotics and ai and work, sap analyst"
    ],
    "Finance / Banking": [
        "finance", "banking", "consulting (m&a)"
    ],
    "Engineering / Tech Industry": [
        "energy systems engineering", "capital equipment",
        "air-conditioning", "automotive", "aviation"
    ],
    "Social Sciences / Psychology": [
        "social sciences", "psychology"
    ],
    "Fishing / Fish Industry": [
        "fishing", "fish industry"
    ],
    "Business / Management": [
        "business administration", "procurement"
    ],
    "Media / Creative Arts": [
        "film", "marketing", "media and communication management in music"
    ],
    "Fashion": [
        "fashion"
    ],
    "Chemical Industry": [
        "chemical industry"
    ]

}

# Normalize input column
df_completed["DemQ04_lower"] = df_completed["DemQ04"].str.strip().str.lower()

# Reverse the map to term → category
reverse_map = {
    term: category
    for category, terms in field_map.items()
    for term in terms
}

'''
# Map cleaned values
df_completed["DemQ04_clean"] = df_completed["DemQ04_lower"].map(reverse_map)

# Show cleaned percentage distribution
demq04_cleaned_distribution = df_completed["DemQ04_clean"].value_counts(normalize=True, dropna=True) * 100
#print("Cleaned field distribution (%):")
#print(demq04_cleaned_distribution)
'''

# --- Replacement for your existing snippet ---

# Map cleaned values
df_completed["DemQ04_clean"] = df_completed["DemQ04_lower"].map(reverse_map)

print("\n## Cleaned Field Distribution ##\n")

# Calculate and print the raw counts, including unclassified fields (NaN)
print("--- Counts ---")
demq04_counts = df_completed["DemQ04_clean"].value_counts(dropna=False)
print(demq04_counts)
print(f"Total participants checked: {demq04_counts.sum()}")


# Calculate and print the percentages, including unclassified fields (NaN)
print("\n--- Percentages ---")
demq04_cleaned_distribution = df_completed["DemQ04_clean"].value_counts(normalize=True, dropna=False) * 100
print(demq04_cleaned_distribution.round(2).astype(str) + '%')
print(f"Total percentage: {demq04_cleaned_distribution.sum().round()}%")

# Compute mean and standard deviation for all numeric columns in df_completed
numeric_stats = df_completed.select_dtypes(include=[np.number]).agg(['mean', 'std']).transpose()

# Save to CSV or print
print(numeric_stats)
numeric_stats.to_csv("Ressources/df_completed_descriptive_stats.csv")

#mappings of non-numericals to numerical

gender_map = {"Male" : 0, "Female" : 1}

time_map = {"No" : 0,
            "Half-time" : 1,
            "Full-time" : 2}

hierarchy_map = {"Other" : 0,
                 "Entry-level" : 0,
                 "Mid-level" : 1,
                 "Senior-level" : 2,
                 "Executive" : 3}

teamSize_map = {"1 (I work alone)" : 0,
                "2-5" : 1,
                "6-10" : 2,
                "11-20" : 3,
                "21+" : 4,
                 }

#scoial = Knowledge Sharing Behavior Scale
social_map = {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "always": 5
        }

frequency_map = {
        "Never": 1,
        "Once a month": 2,
        "Multiple times a month": 3,
        "Once a week": 4,
        "Multiple times a week": 5,
        "Every day": 6,
        "Multiple times a day": 7,
        "Every time I need help": 8
        }

#responsibility_map = {"Very low" : 0,}

#df_completed.replace(r'^\s*(\d+)\s*-.*$', r'\1', regex=True, inplace=True)
#df = df_completed.apply(pd.to_numeric, errors='ignore')

df = df_completed.copy()

df["DemQ01"] = df["DemQ01"].map(gender_map)

for col in ["DemQ07[SQ001]", "DemQ07[SQ002]"]:
    df[col] = df[col].map(time_map)


df["DemQ05"] = df["DemQ05"].map(hierarchy_map)
df["DemQ06"] = df["DemQ06"].map(teamSize_map)

# maps all the scales which entail number but also indicator (i.e. "1 - Not at all") to numerical, here from first column
# which is part of reponsitbility to last column of AI Literacy scale

start = df.columns.get_loc("DemQ51[SQ001]")
end = df.columns.get_loc("LitQ01[SQ015]")

columns = df.columns[start:end+1]

for col in columns:
    df[col].replace(r'^\s*(\d+)\s*-.*$', r'\1', regex=True, inplace=True)
    df[col] = df[col].apply(pd.to_numeric, errors='ignore')

df["UseQ01[SQ001]"] = df["UseQ01[SQ001]"].map(frequency_map)

# maps all the scales which entail number but also indicator (i.e. "1 - Not at all") to numerical, here from first column
# which is part of task complexity to last column of infromal learning scale

start = df.columns.get_loc("GTCQ01[SQ001]")
end = df.columns.get_loc("ILQ01[SQ015]")

columns = df.columns[start:end+1]

for col in columns:
    df[col].replace(r'^\s*(\d+)\s*-.*$', r'\1', regex=True, inplace=True)
    df[col] = df[col].apply(pd.to_numeric, errors='ignore')

# maps all the scales which entail number but also indicator (i.e. "1 - Not at all") to numerical, here from first column
# which is part of Knowledge Sharing Behavior Scale to last column of KSBS

start = df.columns.get_loc("WrittQ01[SQ001]")
end = df.columns.get_loc("CommQ01[SQ007]")

columns = df.columns[start:end+1]

for col in columns:
    df[col] = df[col].map(social_map)

# maps all the scales which entail number but also indicator (i.e. "1 - Not at all") to numerical, here from first column
# which is part of perceived ease of use scale to last column of perceived usefulness scale

start = df.columns.get_loc("PEUQ01[SQ001]")
end = df.columns.get_loc("PUQ01[SQ007]")

columns = df.columns[start:end+1]


for col in columns:
    df[col].replace(r'^\s*(\d+)\s*-.*$', r'\1', regex=True, inplace=True)
    df[col] = df[col].apply(pd.to_numeric, errors='ignore')

'''
# DemQ03: predefined categories (e.g., Employee, Student, Academic)
demq03_distribution = df_completed["DemQ03"].value_counts(normalize=True, dropna=True) * 100
'''
#print("Main occupation distribution (%):")
#print(demq03_distribution)

# --- Replacement for your existing snippet ---

print("## Main Occupation Distribution ##\n")

# Calculate and print the raw counts, including non-responses (NaN)
print("--- Counts ---")
demq03_counts = df_completed["DemQ03"].value_counts(dropna=False)
print(demq03_counts)
print(f"Total participants checked: {demq03_counts.sum()}")


# Calculate and print the percentages, including non-responses (NaN)
print("\n--- Percentages ---")
demq03_distribution = df_completed["DemQ03"].value_counts(normalize=True, dropna=False) * 100
print(demq03_distribution.round(2).astype(str) + '%')
print(f"Total percentage: {demq03_distribution.sum().round()}%")

# Calculate and print the gender distribution (counts)
print("\n## Gender Distribution (Count) ##")
gender_counts = df_completed["DemQ01"].value_counts(dropna=False)
print(gender_counts)

# Calculate and print the gender distribution (percentage)
print("\n## Gender Distribution (Percentage) ##")
gender_percentages = df_completed["DemQ01"].value_counts(normalize=True, dropna=False) * 100
print(gender_percentages.round(2).astype(str) + '%')

# identify whihc columns give qualitative answers which first will be neglected in the correlation analysis
qualitative_columns = [
    "DemQ03", "DemQ04", "UseQ02", "OpenQ01", "OpenQ02", "DemQ05[other]",
    "G10Q28"
]

# drop all qualitative answers from data frame
df_quant = df.drop(columns=qualitative_columns, errors='ignore')

# consolidaten of two columns, where the first was years in postion and second was months in postion
df_quant["timeInPosition"] = df_quant["DemG055[SQ001]"] + df_quant["DemG055[SQ002]"] / 12
#print(df_quant)


# consolidation of the individual quesiton items into their overlaying construct

# constructs as found in the papers
constructsDict_quant = {"gender" : "DemQ01",
                  "age" : "DemQ02",
                  #"occupation" : "DemQ03",
                  #"field" : "DemQ04",
                  "studyPFtime" : "DemQ07[SQ001]",
                  "workPFtime" : "DemQ07[SQ002]",
                  "yearsWorkExp" : "G01Q31",
                  "timeInPosition" : "timeInPosition",
                  "hierarchy" : "DemQ05",
                        "sizeTeam" : "DemQ06",
                        "responsibility" : ["DemQ51[SQ001]","DemQ51[SQ002]", "DemQ51[SQ003]",
                                      "DemQ51[SQ004]", "DemQ51[SQ005]", "DemQ51[SQ006]"],
                        "applyAI" : ["LitQ01[SQ001]", "LitQ01[SQ002]", "LitQ01[SQ003]",
                               "LitQ01[SQ004]", "LitQ01[SQ005]", "LitQ01[SQ006]"],
                        "understandAI" : ["LitQ01[SQ007]", "LitQ01[SQ008]", "LitQ01[SQ009]",
                                    "LitQ01[SQ010]", "LitQ01[SQ011]", "LitQ01[SQ012]"],
                        "detectAI" : ["LitQ01[SQ013]", "LitQ01[SQ014]", "LitQ01[SQ015]"],
                        "frequency" : "UseQ01[SQ001]",
                        "llmTaskComplexity" : ["GTCQ01[SQ001]", "GTCQ01[SQ002]", "GTCQ01[SQ003]", "GTCQ01[SQ004]",
                                        "GTCQ01[SQ005]"],
                        "colleagueTaskComplexity" : ["GTCQ02[SQ001]", "GTCQ02[SQ002]", "GTCQ02[SQ003]",
                                               "GTCQ02[SQ004]", "GTCQ02[SQ005]"],
                        "ownIdeas" : "ILQ01[SQ001]",
                        "modelLearning" : ["ILQ01[SQ002]", "ILQ01[SQ007]", "ILQ01[SQ009]"],
                        "directFeedback" : ["ILQ01[SQ003]", "ILQ01[SQ010]"],
                        "vicariousFeedback" : ["ILQ01[SQ004]", "ILQ01[SQ011]"],
                        "anticipatoryReflection" : ["ILQ01[SQ005]", "ILQ01[SQ012]"],
                        "subsequentReflection" : ["ILQ01[SQ006]", "ILQ01[SQ013]"],
                        "intrinsicIntend" : "ILQ01[SQ008]",
                        "extrinsicIntend" : ["ILQ01[SQ014]", "ILQ01[SQ015]"],
                        "writtenContribution" : ["WrittQ01[SQ001]", "WrittQ01[SQ002]"],
                        "orgComms" : ["OrgQ01[SQ001]", "OrgQ01[SQ002]", "OrgQ01[SQ003]", "OrgQ01[SQ004]",
                                "OrgQ01[SQ005]", "OrgQ01[SQ006]", "OrgQ01[SQ007]", "OrgQ01[SQ008]"],
                        "communContr" : ["CommQ01[SQ001]", "CommQ01[SQ002]", "CommQ01[SQ003]",
                                   "CommQ01[SQ004]", "CommQ01[SQ005]", "CommQ01[SQ006]", "CommQ01[SQ007]"],
                        "pEU" : ["PEUQ01[SQ001]", "PEUQ01[SQ002]", "PEUQ01[SQ003]",
                                    "PEUQ01[SQ004]", "PEUQ01[SQ005]", "PEUQ01[SQ006]"],
                        "pUse" : ["PUQ01[SQ001]", "PUQ01[SQ002]", "PUQ01[SQ003]", "PUQ01[SQ004]",
                            "PUQ01[SQ005]", "PUQ01[SQ006]", "PUQ01[SQ007]"],
                        "interviewtime" : "interviewtime"}

# constructs that consolidates informal learning q_items into "non-llm" and llm inforaml learning construct
constructsDict_quant_fdbk = {"gender" : "DemQ01",
                  "age" : "DemQ02",
                  #"occupation" : "DemQ03",
                  #"field" : "DemQ04",
                  "studyPFtime" : "DemQ07[SQ001]",
                  "workPFtime" : "DemQ07[SQ002]",
                  "yearsWorkExp" : "G01Q31",
                  "timeInPosition" : "timeInPosition",
                  "hierarchy" : "DemQ05",
                        "sizeTeam" : "DemQ06",
                        "responsibility" : ["DemQ51[SQ001]","DemQ51[SQ002]", "DemQ51[SQ003]",
                                      "DemQ51[SQ004]", "DemQ51[SQ005]", "DemQ51[SQ006]"],
                        "applyAI" : ["LitQ01[SQ001]", "LitQ01[SQ002]", "LitQ01[SQ003]",
                               "LitQ01[SQ004]", "LitQ01[SQ005]", "LitQ01[SQ006]"],
                        "understandAI" : ["LitQ01[SQ007]", "LitQ01[SQ008]", "LitQ01[SQ009]",
                                    "LitQ01[SQ010]", "LitQ01[SQ011]", "LitQ01[SQ012]"],
                        "detectAI" : ["LitQ01[SQ013]", "LitQ01[SQ014]", "LitQ01[SQ015]"],
                        "frequency" : "UseQ01[SQ001]",
                        "llmTaskComplexity" : ["GTCQ01[SQ001]", "GTCQ01[SQ002]", "GTCQ01[SQ003]", "GTCQ01[SQ004]",
                                        "GTCQ01[SQ005]"],
                        "colleagueTaskComplexity" : ["GTCQ02[SQ001]", "GTCQ02[SQ002]", "GTCQ02[SQ003]",
                                               "GTCQ02[SQ004]", "GTCQ02[SQ005]"],
                        "IL_Non_LLM" : ["ILQ01[SQ001]", "ILQ01[SQ005]", "ILQ01[SQ006]", "ILQ01[SQ008]",
                                        "ILQ01[SQ009]", "ILQ01[SQ010]", "ILQ01[SQ011]", "ILQ01[SQ012]",
                                        "ILQ01[SQ013]", "ILQ01[SQ014]", "ILQ01[SQ015]"],
                        "IL_LLM" : ["ILQ01[SQ002]", "ILQ01[SQ003]", "ILQ01[SQ004]", "ILQ01[SQ007]"],
                        "writtenContribution" : ["WrittQ01[SQ001]", "WrittQ01[SQ002]"],
                        "orgComms" : ["OrgQ01[SQ001]", "OrgQ01[SQ002]", "OrgQ01[SQ003]", "OrgQ01[SQ004]",
                                "OrgQ01[SQ005]", "OrgQ01[SQ006]", "OrgQ01[SQ007]", "OrgQ01[SQ008]"],
                        "communContr" : ["CommQ01[SQ001]", "CommQ01[SQ002]", "CommQ01[SQ003]",
                                   "CommQ01[SQ004]", "CommQ01[SQ005]", "CommQ01[SQ006]", "CommQ01[SQ007]"],
                        "pEU" : ["PEUQ01[SQ001]", "PEUQ01[SQ002]", "PEUQ01[SQ003]",
                                    "PEUQ01[SQ004]", "PEUQ01[SQ005]", "PEUQ01[SQ006]"],
                        "pUse" : ["PUQ01[SQ001]", "PUQ01[SQ002]", "PUQ01[SQ003]", "PUQ01[SQ004]",
                            "PUQ01[SQ005]", "PUQ01[SQ006]", "PUQ01[SQ007]"],
                        "interviewtime" : "interviewtime"}

# consolidates all q_items of items in the AI Literacy scale into one construct, as paper deemed the underlying
# constructs apply, understand and detect AI have a high factor load to overlaying construct "AI Literarcy"
constructsDict_quant_AIlit = {"gender" : "DemQ01",
                  "age" : "DemQ02",
                  #"occupation" : "DemQ03",
                  #"field" : "DemQ04",
                  "studyPFtime" : "DemQ07[SQ001]",
                  "workPFtime" : "DemQ07[SQ002]",
                  "yearsWorkExp" : "G01Q31",
                  "timeInPosition" : "timeInPosition",
                  "hierarchy" : "DemQ05",
                        "sizeTeam" : "DemQ06",
                        "responsibility" : ["DemQ51[SQ001]","DemQ51[SQ002]", "DemQ51[SQ003]",
                                      "DemQ51[SQ004]", "DemQ51[SQ005]", "DemQ51[SQ006]"],
                        "AI_Lit" : ["LitQ01[SQ001]", "LitQ01[SQ002]", "LitQ01[SQ003]",
                                    "LitQ01[SQ004]", "LitQ01[SQ005]", "LitQ01[SQ006]",
                                    "LitQ01[SQ007]", "LitQ01[SQ008]", "LitQ01[SQ009]",
                                    "LitQ01[SQ010]", "LitQ01[SQ011]", "LitQ01[SQ012]",
                                    "LitQ01[SQ013]", "LitQ01[SQ014]", "LitQ01[SQ015]"],
                        "frequency" : "UseQ01[SQ001]",
                        "llmTaskComplexity" : ["GTCQ01[SQ001]", "GTCQ01[SQ002]", "GTCQ01[SQ003]", "GTCQ01[SQ004]",
                                        "GTCQ01[SQ005]"],
                        "colleagueTaskComplexity" : ["GTCQ02[SQ001]", "GTCQ02[SQ002]", "GTCQ02[SQ003]",
                                               "GTCQ02[SQ004]", "GTCQ02[SQ005]"],
                        "IL_Non_LLM" : ["ILQ01[SQ001]", "ILQ01[SQ005]", "ILQ01[SQ006]", "ILQ01[SQ008]",
                                        "ILQ01[SQ009]", "ILQ01[SQ010]", "ILQ01[SQ011]", "ILQ01[SQ012]",
                                        "ILQ01[SQ013]", "ILQ01[SQ014]", "ILQ01[SQ015]"],
                        "IL_LLM" : ["ILQ01[SQ002]", "ILQ01[SQ003]", "ILQ01[SQ004]", "ILQ01[SQ007]"],
                        "writtenContribution" : ["WrittQ01[SQ001]", "WrittQ01[SQ002]"],
                        "orgComms" : ["OrgQ01[SQ001]", "OrgQ01[SQ002]", "OrgQ01[SQ003]", "OrgQ01[SQ004]",
                                "OrgQ01[SQ005]", "OrgQ01[SQ006]", "OrgQ01[SQ007]", "OrgQ01[SQ008]"],
                        "communContr" : ["CommQ01[SQ001]", "CommQ01[SQ002]", "CommQ01[SQ003]",
                                   "CommQ01[SQ004]", "CommQ01[SQ005]", "CommQ01[SQ006]", "CommQ01[SQ007]"],
                        "pEU" : ["PEUQ01[SQ001]", "PEUQ01[SQ002]", "PEUQ01[SQ003]",
                                    "PEUQ01[SQ004]", "PEUQ01[SQ005]", "PEUQ01[SQ006]"],
                        "pUse" : ["PUQ01[SQ001]", "PUQ01[SQ002]", "PUQ01[SQ003]", "PUQ01[SQ004]",
                            "PUQ01[SQ005]", "PUQ01[SQ006]", "PUQ01[SQ007]"],
                        "interviewtime" : "interviewtime"}

# consolidates all q_items from the KSBS into one, however I want to align with Rosemarie first if that is smart,
# changes the correlation a lot and reduces interpretability
constructsDict_quant_KS = {"gender" : "DemQ01",
                  "age" : "DemQ02",
                  #"occupation" : "DemQ03",
                  #"field" : "DemQ04",
                  "studyPFtime" : "DemQ07[SQ001]",
                  "workPFtime" : "DemQ07[SQ002]",
                  "yearsWorkExp" : "G01Q31",
                  "timeInPosition" : "timeInPosition",
                  "hierarchy" : "DemQ05",
                        "sizeTeam" : "DemQ06",
                        "responsibility" : ["DemQ51[SQ001]","DemQ51[SQ002]", "DemQ51[SQ003]",
                                      "DemQ51[SQ004]", "DemQ51[SQ005]", "DemQ51[SQ006]"],
                        "AI_Lit" : ["LitQ01[SQ001]", "LitQ01[SQ002]", "LitQ01[SQ003]",
                                    "LitQ01[SQ004]", "LitQ01[SQ005]", "LitQ01[SQ006]",
                                    "LitQ01[SQ007]", "LitQ01[SQ008]", "LitQ01[SQ009]",
                                    "LitQ01[SQ010]", "LitQ01[SQ011]", "LitQ01[SQ012]",
                                    "LitQ01[SQ013]", "LitQ01[SQ014]", "LitQ01[SQ015]"],
                        "frequency" : "UseQ01[SQ001]",
                        "llmTaskComplexity" : ["GTCQ01[SQ001]", "GTCQ01[SQ002]", "GTCQ01[SQ003]", "GTCQ01[SQ004]",
                                        "GTCQ01[SQ005]"],
                        "colleagueTaskComplexity" : ["GTCQ02[SQ001]", "GTCQ02[SQ002]", "GTCQ02[SQ003]",
                                               "GTCQ02[SQ004]", "GTCQ02[SQ005]"],
                        "IL_Non_LLM" : ["ILQ01[SQ001]", "ILQ01[SQ005]", "ILQ01[SQ006]", "ILQ01[SQ008]",
                                        "ILQ01[SQ009]", "ILQ01[SQ010]", "ILQ01[SQ011]", "ILQ01[SQ012]",
                                        "ILQ01[SQ013]", "ILQ01[SQ014]", "ILQ01[SQ015]"],
                        "IL_LLM" : ["ILQ01[SQ002]", "ILQ01[SQ003]", "ILQ01[SQ004]", "ILQ01[SQ007]"],
                        "K_Sharing" : ["WrittQ01[SQ001]", "WrittQ01[SQ002]", "OrgQ01[SQ001]", "OrgQ01[SQ002]",
                                       "OrgQ01[SQ003]", "OrgQ01[SQ004]", "OrgQ01[SQ005]", "OrgQ01[SQ006]",
                                       "OrgQ01[SQ007]", "OrgQ01[SQ008]", "CommQ01[SQ001]", "CommQ01[SQ002]",
                                       "CommQ01[SQ003]", "CommQ01[SQ004]", "CommQ01[SQ005]", "CommQ01[SQ006]",
                                       "CommQ01[SQ007]"],
                        "pEU" : ["PEUQ01[SQ001]", "PEUQ01[SQ002]", "PEUQ01[SQ003]",
                                    "PEUQ01[SQ004]", "PEUQ01[SQ005]", "PEUQ01[SQ006]"],
                        "pUse" : ["PUQ01[SQ001]", "PUQ01[SQ002]", "PUQ01[SQ003]", "PUQ01[SQ004]",
                            "PUQ01[SQ005]", "PUQ01[SQ006]", "PUQ01[SQ007]"],
                        "interviewtime" : "interviewtime"}

# create a new df to store construct level scores
construct_scores = pd.DataFrame(index=df_quant.index)

# iterates over all items in the selected construct dictionary
for construct, cols in constructsDict_quant_AIlit.items():
    #checks if a constructs has multiple items, thus multiple colums
    if isinstance(cols, list):
        # ensure that the columns in new df construct_scores exists in original df df_quant
        valid_cols = [col for col in cols if col in df_quant.columns]
        # checks of valid_cols has at least one element
        if valid_cols:
            #do cronbach's alpha test
            alpha = cronbach_alpha(df_quant[valid_cols])
            print(f"Cronbach's alpha for '{construct}': {alpha:.3f}")
            #do mean
            construct_scores[construct] = df_quant[valid_cols].mean(axis=1, skipna=True)
    # in case construct has only one item like "gender" : "DemQ01"
    else:
        if cols in df_quant.columns:
            construct_scores[construct] = df_quant[cols]

from itertools import combinations

#check individual items of llmTaskComplexity
subset = df_quant[constructsDict_quant_AIlit["llmTaskComplexity"]].dropna()
correlation_matrix = subset.corr()
print(correlation_matrix)

for col in subset.columns:
    alpha_if_dropped = cronbach_alpha(subset.drop(columns=col))
    print(f"Alpha without {col}: {alpha_if_dropped:.3f}")

#check the two items of written contribution
subset = df_quant[constructsDict_quant_AIlit["writtenContribution"]].dropna()
corr = subset.corr().iloc[0, 1]
print(f"Correlation between the two items: {corr:.3f}")

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import textwrap

# --- Add this code to your script after df_completed is created ---



# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    """Cleans, tokenizes, lemmatizes, and removes stopwords from a text."""
    # 1. Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Lemmatize and remove stopwords
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return clean_tokens


# --- THE NEW, CORRECTED ANALYSIS FUNCTION ---

def analyze_and_get_context(data_series, title, custom_stopwords=None, top_n=7):
    """
    Performs frequency analysis on a text series and retrieves the original
    context for the most frequent terms.
    """
    print(f"\n{'=' * 80}\nANALYSIS FOR: {title}\n{'=' * 80}")

    # 1. Prepare data and stopwords for this specific analysis
    local_stop_words = stop_words.copy()
    if custom_stopwords:
        local_stop_words.update(custom_stopwords)

    text_data = data_series.dropna().astype(str)

    # 2. Process each response, linking original to tokens
    processed_responses = []
    for response in text_data:
        # Re-run process_text with the specific stopwords for this analysis
        text = response.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        clean_tokens = [
            lemmatizer.lemmatize(word) for word in tokens
            if word not in local_stop_words and len(word) > 2
        ]
        if clean_tokens:
            processed_responses.append({
                'original': response.strip(),
                'tokens': clean_tokens
            })

    # 3. Get frequency counts based on DOCUMENTS, not total tokens
    # <-- THIS IS THE KEY CHANGE
    # We count how many documents/responses contain each term at least once.
    document_tokens = [set(resp['tokens']) for resp in processed_responses]
    document_frequency_list = [token for doc_set in document_tokens for token in doc_set]

    if not document_frequency_list:
        print("No significant terms found after processing.")
        return

    word_counts = Counter(document_frequency_list)  # This now counts responses
    top_terms = [word for word, count in word_counts.most_common(top_n)]

    print(f"Top {top_n} themes found: {', '.join(top_terms)}\n")

    # 4. Build the context map (this part was already correct)
    context_map = {term: [] for term in top_terms}
    for resp in processed_responses:
        present_tokens = set(resp['tokens'])
        for term in top_terms:
            if term in present_tokens:
                context_map[term].append(resp['original'])

    # 5. Print the results - the numbers will now match
    for term in top_terms:
        # The count now correctly reflects the number of responses
        count = word_counts[term]
        print(f"\n--- Term: '{term}' (Found in {count} responses) ---")
        for i, sentence in enumerate(context_map[term]):
            wrapped_sentence = textwrap.fill(sentence, width=75, initial_indent="  ", subsequent_indent="  ")
            print(f"{i + 1}. {wrapped_sentence}")


# --- RUN THE ANALYSIS FOR EACH OPEN-ENDED QUESTION ---

# For "Personal Use Cases"
custom_stopwords_use = {'llm', 'llms', 'use', 'using', 'chatgpt', 'like', 'also', 'get', 'etc', 'ai', 'text'}
analyze_and_get_context(
    data_series=df_completed['UseQ02'],
    title="Personal Use Cases (UseQ02)",
    custom_stopwords=custom_stopwords_use,
    top_n=7 # You can change how many top terms you want to see
)

# For "Impressive/Useful Examples"
custom_stopwords_impressive = {'llm', 'llms', 'example', 'examples', 'impressed', 'impressive', 'useful', 'found', 'seen', 'ai', 'take', 'good'}
analyze_and_get_context(
    data_series=df_completed['OpenQ01'],
    title="Impressive/Useful Examples (OpenQ01)",
    custom_stopwords=custom_stopwords_impressive,
    top_n=7
)

# For "Concerning Examples"
custom_stopwords_concerning = {'llm', 'llms', 'example', 'examples', 'concerning', 'concern', 'worried', 'scary', 'ai', 'without', 'always', 'used'}
analyze_and_get_context(
    data_series=df_completed['OpenQ02'],
    title="Concerning Examples (OpenQ02)",
    custom_stopwords=custom_stopwords_concerning,
    top_n=7
)

# Reorder the columns: move 'timeInPosition' right after 'yearsWorkExp'
cols = construct_scores.columns.tolist()
if "yearsWorkExp" in cols and "timeInPosition" in cols:
    cols.remove("timeInPosition")
    idx = cols.index("yearsWorkExp")
    cols.insert(idx + 1, "timeInPosition")
    construct_scores = construct_scores[cols]

# Select only numeric constructs (to be safe)
construct_scores_numeric = construct_scores.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = construct_scores_numeric.corr(method="pearson")

# Display or export
#print(correlation_matrix)
correlation_matrix.to_csv("Ressources/construct_correlation_matrix_10062025.csv")



# Identify strongly correlated pairs (absolute correlation > 0.6 and not 1.0)
threshold = 0.344
corr_pairs = correlation_matrix.unstack()
strong_corr = corr_pairs[(abs(corr_pairs) > threshold) & (abs(corr_pairs) < 1.0)]
strong_corr = strong_corr.drop_duplicates().sort_values(ascending=False)

# Convert to DataFrame and save
strong_corr_df = strong_corr.reset_index()
strong_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
strong_corr_df.to_csv("Ressources/strong_correlations_10062025.csv", index=False)

# --- Add this code block to the end of your script ---

print(f"\n{'='*80}\nSIMPLE FREQUENCY COUNT FOR: Concerning Examples (OpenQ02)\n{'='*80}")

# 1. Define the custom stopwords for this specific analysis
custom_stopwords_concerning = {'llm', 'llms', 'example', 'examples', 'concerning', 'concern', 'worried', 'scary', 'ai'}
local_stop_words = stop_words.copy()
local_stop_words.update(custom_stopwords_concerning)

# 2. Select the data and drop any empty responses
data_series = df_completed['OpenQ02']
text_data = data_series.dropna().astype(str)

# 3. Process each response to get cleaned tokens
processed_tokens_list = []
for response in text_data:
    # We define the text processing steps here directly
    text = response.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in local_stop_words and len(word) > 2
    ]
    if clean_tokens:
        processed_tokens_list.append(clean_tokens)

# 4. Calculate the document frequency (how many responses contain each word)
document_tokens = [set(resp_tokens) for resp_tokens in processed_tokens_list]
document_frequency_list = [token for doc_set in document_tokens for token in doc_set]
word_counts = Counter(document_frequency_list)

# 5. Print the top 20 terms in the desired "word: count" format
print("\nTop 20 terms and their frequency of occurrence in responses:\n")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")