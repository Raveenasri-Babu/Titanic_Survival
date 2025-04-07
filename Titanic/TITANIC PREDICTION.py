#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd  
df = pd.read_csv(r"C:\!!!TITANIC\Titanic.csv")  
df.head()


# In[6]:


df.isnull().sum()
df.describe()
df.info()


# In[1]:


df.describe()


# In[8]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['PortEmbarked'].fillna(df['PortEmbarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.head()


# In[9]:


df.Name


# In[10]:


df['Title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=False)
df['Title'].value_counts()
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                   'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                   'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].map({ 'Mr': 1, 'Miss': 2,'Mrs': 3, 'Master': 4,'Rare': 5})
df.head()


# In[11]:


df = pd.read_csv(r"C:\!!!TITANIC\Titanic.csv")


# In[12]:


df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 
    'Rare')
df['Title'] = df['Title'].map({ 'Mr': 1,'Miss': 2, 'Mrs': 3,'Master': 4,'Rare': 5})
df.drop(columns=['Name'], inplace=True)
df.head()


# In[13]:


df.drop(columns=['Ticket'], inplace=True, errors='ignore')
df.drop(columns=['Cabin'], inplace=True, errors='ignore')
df.head()


# In[14]:


from sklearn.preprocessing import LabelEncoder

def label_encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df
df = label_encode(df, ['Sex', 'PortEmbarked'])


# In[11]:


df.head()


# In[15]:


import matplotlib.pyplot as plt
import pandas as pd

#Gender
gender_counts = df['Sex'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

# age groups
bins = [0, 12, 18, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
agegroup_counts = df['AgeGroup'].value_counts().sort_index()
colors = ['#ffcccb', '#add8e6', '#90ee90', '#d3d3d3']  # red,blue,green,grey
explode = [0, 0, 0.1, 0]  
plt.figure(figsize=(6, 6))
plt.pie(agegroup_counts,  labels=agegroup_counts.index,  autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
plt.title('Age Group Distribution')
plt.axis('equal')
plt.show()

#PassengerClass
pclass_counts = df['PassengerClass'].value_counts().sort_index()
labels = ['1st Class', '2nd Class', '3rd Class']
plt.figure(figsize=(6, 6))
plt.pie(pclass_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'silver', 'lightblue'])
plt.title('Passenger Class Distribution')
plt.axis('equal')
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Gender vs Survival Rate
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='Sex', y='Survived', ci=None, palette='pastel')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.ylim(0, 1)
plt.show()

# Age Group vs Survival Rate
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='AgeGroup', y='Survived', ci=None, palette='muted')
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.ylim(0, 1)
plt.show()

# Passenger Class vs Survival Rate
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='PassengerClass', y='Survived', ci=None, palette='Set2')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.ylim(0, 1)
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Create fare bins
fare_bins = [0, 10, 30, 50, 100, 600]
fare_labels = ['0-10', '10-30', '30-50', '50-100', '100+']
df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels)

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='FareGroup', y='Survived', ci=None, palette='pastel')
plt.title('Survival Rate by Fare Group')
plt.xlabel('Fare Range ($)')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()


# In[18]:


df['FamilySize'] = df['Siblings'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df[['Siblings','FamilySize', 'IsAlone']].head(10)


# In[20]:


df.head()


# In[29]:


x['Age'] = x['Age'].fillna(x['Age'].median())
x['Fare'] = x['Fare'].fillna(x['Fare'].median())

features = ['PassengerId', 'Sex', 'Age', 'Fare', 'PortEmbarked', 'Title', 'FamilySize', 'IsAlone']
from sklearn.model_selection import train_test_split
x = df[features]
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[24]:


df.head()


# In[27]:


print(x.isnull().sum())


# In[31]:


x = df[features].copy()
x['Age'] = x['Age'].fillna(x['Age'].median())
x['Fare'] = x['Fare'].fillna(x['Fare'].median())


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:




