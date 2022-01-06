#importing liberaries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pickle

#reading and print dataset
df = pd.read_csv("Iris.csv")
print("\nOur Dataset Is:- \n",df.head(10))
print("\nShape Of Dataset Is = ",df.shape)

#droping unwanted column
df = df.drop(columns="Id")
print("\nAfter Droping 'Id' column Dataset will be:- \n",df.head(10))
print("\nShape after droping a column = ",df.shape)

#check is there any null value have or not
print("\nCheck how many null value in our dataset have:- \n",df.isnull().sum())
sns.heatmap(df.isnull(), cmap = "viridis")
plt.title("Check Null Values")
plt.show()

#knoowing data
print("\nStatistical value of dataset :- \n",df.describe())
print("\nSeparated count of species:- \n",df["Species"].value_counts())

#checking correlation between independence variables
sns.heatmap(df.corr(), annot = True)
plt.title("Correction between independence variable")
plt.show()

sns.countplot("Species", data= df)
plt.title("Countplot For Species")
plt.show()

#separating independent and dependent varaiables
x = df.drop(columns= "Species")
y = df["Species"]

#train & test spliting
x_train ,x_test, y_train, y_test = train_test_split(x,y, test_size=.20, random_state=10)

#creating decision tree model
df_model = DecisionTreeClassifier(max_depth= 6)
df_model.fit(x_train, y_train)

#making prediction through our model
"""y_predict = df_model.predict([[6.2,2.9,4.3,1.3]])
print(y_predict)
print("\nAccuray Score of our model is :- ")"""

#Ploting decision tree
plot_tree(df_model, filled=True)
plt.title("Decision Tree For Iris Dataset")
plt.show()

#Completed By Tushar Sonp
print(x_test)

#save the model ro disk
model = pickle.dump(df_model, open("decision_tree.pkl", "wb"))

#to loading the model to compare the result
model = pickle.load(open("decision_tree.pkl", "rb"))