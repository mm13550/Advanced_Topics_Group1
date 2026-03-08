#Requirements: pandas matplotlib datasets

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

#Load the dataset
ds = load_dataset("ChicagoHAI/CaseSumm")
df = ds['train'].to_pandas()
print(df.head())

#Look at the structure of the dataset
print(df.info())
print(df.describe())

#Check for null values
print(df.isnull().sum())

#Analyze the length of the opinion and syllabus texts
df["opinion_length"] = df["opinion"].apply(len)
df["syllabus_length"] = df["syllabus"].apply(len)

print(df[["opinion_length", "syllabus_length"]].describe())

#Visualize the distribution of opinion lengths
plt.hist(df["opinion_length"], bins=20)
plt.title("Distribution of Opinion Lengths")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.show()

#Visualize the distribution of syllabus lengths
plt.hist(df["syllabus_length"], bins=20)
plt.title("Distribution of Syllabus Lengths")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.show()

#Visualize the relationship between syllabus length and opinion length
plt.scatter(df["syllabus_length"], df["opinion_length"])
plt.xlabel("Syllabus Length")
plt.ylabel("Opinion Length")
plt.title("Summary Length vs Opinion Length")
plt.show()

#Calculate the number of words in the opinion and syllabus texts
df["opinion_word_count"] = df["opinion"].apply(lambda x: len(x.split()))
df["syllabus_word_count"] = df["syllabus"].apply(lambda x: len(x.split()))

#Visualize the distribution of opinion word counts
plt.hist(df["opinion_word_count"], bins=20)
plt.title("Distribution of Opinion Word Counts")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

#Visualize the distribution of syllabus word counts
plt.hist(df["syllabus_word_count"], bins=20)
plt.title("Distribution of Syllabus Word Counts")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
