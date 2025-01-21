
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("phishing.csv")
data.head()

# Information about the dataset
data.info()

data = data.drop(['Index'], axis=1)

data.describe().T

# Correlation heatmap

plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True)
plt.show()

# pairplot for particular features

df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS',
           'AnchorURL', 'WebsiteTraffic', 'class']]
sns.pairplot(data=df, hue="class", corner=True)

# Phishing Count in pie chart

data['class'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.title("Phishing Count")
plt.show()