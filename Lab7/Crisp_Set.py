import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

def classify_petal_length(row):
    if row['petal length (cm)'] < 2.5:
        return 'Setosa'
    elif 2.5 <= row['petal length (cm)'] <= 5.0:
        return 'Versicolor'
    else:
        return 'Virginica'

iris_df['Crisp Set'] = iris_df.apply(classify_petal_length, axis=1)

print(iris_df[['petal length (cm)', 'Crisp Set']])
