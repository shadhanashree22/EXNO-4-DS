# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![322338017-b544c435-1cc1-4bc6-83c9-de2945348808](https://github.com/user-attachments/assets/205a82d7-6373-4171-8e74-5010755763de)

```
data.isnull().sum()
```
 ![322338037-40b1ab98-5a1a-41a1-b943-102b7c4cabed](https://github.com/user-attachments/assets/388aafce-4f91-4cdd-be3c-971b5cafbe17)

```

missing=data[data.isnull().any(axis=1)]
missing
```
![322338066-a5fe88ab-c993-4c97-b249-cffea5a21a54](https://github.com/user-attachments/assets/40717901-65b3-44cf-8f7f-16e182e57e2a)

```

data2=data.dropna(axis=0)
data2
```
![322338086-40a10680-63a6-4f18-87ae-517ceda76ca9](https://github.com/user-attachments/assets/5a8daad0-4024-4643-9686-0e0c76b57a3d)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![322338114-e59ce957-1bdc-4455-97a5-15d66108b864](https://github.com/user-attachments/assets/2645d0ab-24f6-42b8-aa95-8a693d9f8a69)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![322338170-f8435063-835b-4eba-af2e-c46c67ea55e9](https://github.com/user-attachments/assets/a77ae613-9c25-49ca-a758-ad8ab8859430)

```
data2
```

![322338187-c034e83a-8e21-400e-bc40-103e3da86d0e](https://github.com/user-attachments/assets/e4463908-50ad-4768-9430-b408773c7688)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![322338213-f21819e3-a5bd-47e6-b1b7-9bc08b64bed9](https://github.com/user-attachments/assets/cd824f19-eea5-475d-beda-344d7f921dde)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![322338249-8af6f5ce-4d99-4ed6-9371-730aeaa5a56b](https://github.com/user-attachments/assets/50bcecee-400c-48f5-a37c-83ae7f4a2b42)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![322338261-5f31a677-7d30-417a-8044-d5db741cafbf](https://github.com/user-attachments/assets/8afa9840-600a-4b58-9a38-6d3e6b5fa63f)

```
y=new_data['SalStat'].values
print(y)
```
![322338286-f4c779af-4c87-449e-9daa-be5d8d275212](https://github.com/user-attachments/assets/87cc8704-54a0-4b7b-8d56-c9ae5a9ea412)

```
x=new_data[features].values
print(x)
```
![322338321-4154db03-4c87-4b98-a13b-964f19bee9b0](https://github.com/user-attachments/assets/1786ce40-3fc3-470d-a0c0-f7339c593dda)

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![322338343-e5e02520-eb39-436c-ac2e-e43048c1d672](https://github.com/user-attachments/assets/10a574d7-ddd9-4a54-a741-ca23fa2c4dcf)

```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![322338371-a6eedfe3-aedd-4500-958f-6faafd54f464](https://github.com/user-attachments/assets/c625d274-7454-47d9-a93a-d39cc10a9913)

```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![322338387-0e56ff41-2f35-4d01-b479-53547391567b](https://github.com/user-attachments/assets/a36eae1a-e712-4cf5-a1b4-cc5525dd93eb)

```

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![322338405-4af5ed3f-362a-40c6-a438-c89f31584e51](https://github.com/user-attachments/assets/f3c79ca7-74e0-4120-92ca-a492affbcf73)

```

data.shape
```
![322338420-1986f990-26e6-4b42-acfc-b2a6e52f8042](https://github.com/user-attachments/assets/936994a4-8c6f-454d-aa64-f58fcf038dba)

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![322338454-20777b0d-3cdb-4ae9-80e4-1f76ed093191](https://github.com/user-attachments/assets/01ed62c7-e252-4865-bb1a-c188cc572977)

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322338479-6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6](https://github.com/user-attachments/assets/e92fd39c-769f-41f5-b21e-c0f9cbcdf8be)

```

tips.time.unique()
```
![322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6](https://github.com/user-attachments/assets/a5544fe4-d165-448f-a929-9eeb73c2060b)

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![322338518-06365e9f-f51b-4cf6-ab04-8a136726a025](https://github.com/user-attachments/assets/aa6d5456-0580-4c37-bdbb-ba8ad5f40a50)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![322338543-6adc4da7-421c-458f-9ec6-f6158aa6f731](https://github.com/user-attachments/assets/f6590678-3f62-4e78-a831-8183e148f48e)


# RESULT:

         Thus, Feature selection and Feature scaling has been used on thegiven dataset.
