import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rc("font", size=14)
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from IPython.display import display
import src.Constants as Cns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from src.utils.utility_functions import SelectColumnsTransformer

# df_train = pd.read_csv(Cns.TRAIN_RAW_DATA_FOLDER, header=0)
# df_train.columns = data.columns.str.lower()

# data = data.head(20)
df_train = pd.read_csv('data/train.csv')
print('Dataframe shape is %s:' % str(df_train.shape))
print('Dataframe columns are :')
print(df_train.columns)
# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
missing_data = missing_data[missing_data['total'] > 0]
df_train.columns = [col.lower() for col in df_train.columns]

num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_train['age'] = num_imputer(df_train['age'].values.reshape(-1,1))


cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
selector_preprocessing = Pipeline([('selector', SelectColumnsTransformer(cols))])
df_train = selector_preprocessing.fit_transform(X=df_train)

categorical_preprocessing = Pipeline([('ohe', OneHotEncoder())])

# define which transformer applies to which columns
preprocess = ColumnTransformer([
    ('categorical_preprocessing', categorical_preprocessing, ['favorite_color']),
    ('numerical_preprocessing', numerical_preprocessing, ['age'])
])
# define individual transformers in a pipeline
categorical_preprocessing = Pipeline([('ohe', OneHotEncoder())])
numerical_preprocessing = Pipeline([('imputation', SimpleImputer())])

# define which transformer applies to which columns
preprocess = ColumnTransformer([
    ('categorical_preprocessing', categorical_preprocessing, ['favorite_color']),
    ('numerical_preprocessing', numerical_preprocessing, ['age'])
])

# create the final pipeline with preprocessing steps and
# the final classifier step
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('clf', DecisionTreeClassifier())
])

# now fit the pipeline using the whole dataframe
df_features = df[['favorite_color', 'age']]
df_target = df['target']

# call fit on the dataframes
pipeline.fit(df_features, df_target)

# cols = ['product_category_1', 'product_category_2', 'product_category_3']
# data['all_products_cat'] = data[cols].astype('Int16').astype(str).values.tolist()
# data['all_products_cat'] = data['all_products_cat'].apply(lambda x: ','.join([s for s in x if s != '<NA>']))
# data.info()

# data_person = data.copy().sort_values(by=['user_id'])
# # data_person['all_products_category_str'] = data_person['all_products_category'].apply(lambda x: [x])
# data_person = data_person.drop_duplicates(subset=['user_id'], keep='first')
# data_person = data_person.set_index(data_person['user_id'])
# data_person['purchase'] = data.groupby(['user_id'])['purchase'].sum()
# data_person['all_products_cat'] = data.groupby(['user_id'])['all_products_cat'].apply(lambda x: ','.join(x))
# data_person['all_products_cat'] = data_person['all_products_cat'].apply(lambda x: [int(x) for x in x.split(',')])
# data_person['all_products_cat_set'] = data_person['all_products_cat'].apply(lambda x: set(x))
# data_person['all_products_cat_count'] = data_person['all_products_cat'].apply(lambda x: Counter(x))
#
# cat_num_list = []
# for cat_num in set(list(data['product_category_1'])):
#     cat_num_list.append('product_cat_num_%d' % cat_num)
# data_person[cat_num_list] = 0
# for index, row in data_person.iterrows():
#     for k, v in row['all_products_cat_count'].items():
#         data_person.loc[index, 'product_cat_num_%d' % k] = v

# data_person = pd.read_csv('data_person.csv')
# train_cols = ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
#               'Product_Category_1', 'Purchase']

# x_train, x_test, y_train, y_test = train_test_split(data_person.drop('gender', axis=1) , data_person['gender'],
#                                                     test_size=0.15, random_state=120, shuffle=False)
# x_cat = data_person[['age', 'city_category', 'stay_in_current_city_years', 'marital_status']]
# x_cat = data_person[['city_category']]
# enc = OrdinalEncoder(dtype=np.int16).fit(x_cat)
# print(enc.categories_)
# x_cat = enc.transform(x_cat)
# x_cat_train, x_cat_test = train_test_split(x_cat, test_size=0.15, random_state=120, shuffle=False)
#
# x_cont = data_person[['occupation', 'purchase']]
# x_cont_train, x_cont_test = train_test_split(x_cont, test_size=0.15, random_state=120, shuffle=False)
#
# x_multi = data_person[cat_num_list]
# x_multi_train, x_multi_test = train_test_split(x_multi, test_size=0.15, random_state=120, shuffle=False)
#
# x_bin = (x_multi > 0) * 1
# x_bin_train, x_bin_test = train_test_split(x_bin, test_size=0.15, random_state=120, shuffle=False)
#
# y_str = data_person['gender']
# le = LabelEncoder().fit(y_str)
# print(le.classes_)
# y = le.transform(y_str)
# y_train, y_test = train_test_split(y, test_size=0.15, random_state=120, shuffle=False)
#
# from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB, BernoulliNB
#
# ALPHA = .3
# catNB_skl = CategoricalNB(alpha=ALPHA)
# catNB_skl.fit(x_cat_train, y_train)
#
# contNB_skl = GaussianNB()
# contNB_skl.fit(x_cont_train, y_train)
#
# multiNB_skl = MultinomialNB()
# multiNB_skl.fit(x_multi_train, y_train)
#
# binNB_skl = BernoulliNB()
# binNB_skl.fit(x_bin_train, y_train)
#
# # predictions = clf.predict(X)
# cat_prob = catNB_skl.predict_log_proba(x_cat_test)
# cont_prob = contNB_skl.predict_log_proba(x_cont_test)
# multi_prob = multiNB_skl.predict_log_proba(x_multi_test)
# bin_prob = binNB_skl.predict_log_proba(x_bin_test)
#
# pred_class = np.argmax(cat_prob + cont_prob + multi_prob, axis=1)
# print(f"Sklearn accuracy on train data: {sum(y_test == pred_class) / len(y_test) * 100:.1f}%")
# print(f"Sklearn accuracy on test data: {100*sum(predictions == pred_sklearn)/len(predictions):.2f}%" )
#


# print('aaaa')
# from sklearn.datasets import make_classification
#
# nb_samples = 300
# X, Y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0)
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.model_selection import train_test_split
#
# plt.scatter(X[:, 0], X[:, 1], s=Y * 15, marker='o', c='r')
# plt.scatter(X[:, 0], X[:, 1], s=((Y * -1) + 1) * 15, marker='o', c='b')
# plt.show()
#
# Y = (Y > 0) * 1
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
#
# bnb = BernoulliNB()
# bnb.fit(X_train, Y_train)
# bnb.score(X_test, Y_test)
# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# bnb.predict(data)
#
# from sklearn.feature_extraction import DictVectorizer
#
# data = [{'house': 100, 'street': 50, 'shop': 25, 'car': 100, 'tree': 20},
#         {'house': 5, 'street': 5, 'shop': 0, 'car': 10, 'tree': 500, 'river': 1}]
#
# dv = DictVectorizer(sparse=False)
# X = dv.fit_transform(data)
# Y = np.array([1, 0])
# from sklearn.naive_bayes import MultinomialNB
#
# mnb = MultinomialNB()
# mnb.fit(X, Y)

if __name__ == '__main__':
    print('')
