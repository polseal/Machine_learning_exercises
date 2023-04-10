import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
import warnings
import sklearn.metrics as metrics
np.random.seed(0)

print("1.1", "_" * 50)

wine_data = datasets.load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
target = pd.Series(wine_data.target)

print("Number of examples", df.shape[0])

print("Number of columns", df.shape[1])

print("Attribute names: ", df.columns.values.astype(list))

print("Dataset data types")
print(df.dtypes)

ash_stat = pd.DataFrame(df["ash"].describe()).loc[["mean", "min", "50%", "max"], :]

print(ash_stat)

print("Number of unique iris types:", target.unique().size)

print(target.value_counts())

print("1.2", "_" * 50)

fig = plt.figure(figsize=(6, 6))
height, bins, _ = plt.hist(df["ash"], bins=10)

# print(bins)

plt.xticks(bins, rotation=45)

plt.show()

fig = plt.figure(figsize=(6, 6))

plt.boxplot(df["ash"])

print(
    f"The median 2.36 is denoted as the orange line.The data are slightly skewed to the left: more values are "
    f"concentrated on the right tail.\nThere are also outliers.")

plt.show()

print("2.1", "_" * 50)

df["substracted_phenols"] = df["nonflavanoid_phenols"] - df["total_phenols"]

df["alcohol"] = df["alcohol"] + 1

df['target'] = target

print("2.2", "_" * 50)

df = df.sample(frac=1, random_state=0).reset_index(drop=True)

df_sample = df.sample(n=20, random_state=0)

df_sampleTarget = df_sample.iloc[:, 14]

l = df_sample.shape

l = list(l)

mu, sigma = 0, 1
s = np.random.normal(mu, sigma, l)

new_rows = df_sample + s

new_rows['target'] = pd.Series(df_sampleTarget)

df = df.append(new_rows)

print("Augmented dataset shape:", df.shape)

print("2.3", "_" * 50)

print("I'm using the minmax scaler technique because there's no need to change the set's distribution")
scaler = MinMaxScaler()

df[["ash"]] = scaler.fit_transform(df[["ash"]])

y = df['target'] #preparation for 3.2

X_train1, X_test1, y_train1, y_test1 = train_test_split(df.values, y, test_size=0.15, random_state=0) #preparation for 3.2

print("3.1", "_" * 50)

df1 = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

target1 = pd.Series(wine_data.target)

target1 = target1.apply(lambda x: "not-class-0" if x > 0 else "class-0")

X_training, X_test, y_training, y_test = train_test_split(df1.values, target1, test_size=0.15, random_state=0)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_training, y_training)

predictions = model.predict(X_test)

print("KNN  confusion matrix")
lab = pd.Series(model.classes_)
cm = pd.DataFrame(confusion_matrix(y_test, predictions))
cm.insert(loc=2, column='labels', value=lab)

print(cm, f"Out of the 27 test cases, 25 were identified correctly. TP=8, TN=17.\n KNN classifier is correct in ~93% "
          f"(25/27) or cases (accuracy). KNN is correct in predicting class-0 ~89% (8/9) of time (precision).\n ~89% "
          f"of actual class-0 cases were identified correctly (recall). Precision, recall ~ 94% for not-class-0, "
          f"respectively")

print("KNN classification report")
print(classification_report(y_test, predictions, labels=model.classes_))

print("3.2", "_" * 50)

y_train1 = y_train1.apply(lambda x: "not-class-0" if x > 0 else "class-0")
y_test1 = y_test1.apply(lambda x: "not-class-0" if x > 0 else "class-0")

#model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train1, y_train1)

predictions = model.predict(X_test1)

print("KNN  confusion matrix")
lab = pd.Series(model.classes_)
cm = pd.DataFrame(confusion_matrix(y_test1, predictions))
cm.insert(loc=2, column='labels', value=lab)
print(cm, f"Out of the 30 test cases, 28 were identified correctly. TP=6, TN=22.\n KNN classifier is correct in ~93% "
          f"(28/30) or cases (accuracy). KNN is correct in predicting class-0 ~86% (6/7) of time (precision).\n ~86% "
          f"of actual class-0 cases were identified correctly (recall). Precision, recall ~ 96% for not-class-0, "
          f"respectively")

print("KNN classification report")
print(classification_report(y_test1, predictions, labels=model.classes_))

print("Comparison: a slight loss of precision and recall for class-0, a little gain for not-class-0.\n It could be "
      "caused by the model overfitting on the first dataset.")

print("4.1", "_" * 50)

house_data = datasets.fetch_california_housing()

df = pd.DataFrame(house_data.data, columns=house_data.feature_names)

df['target'] = pd.Series(house_data.target)

missing_values_count = df.isnull().sum()
print("There are no missing values in this dataset")
print(missing_values_count)

correl = df.corr()

correl_np = correl.to_numpy()

print("There is strong positive correlation between MedInc and target (median house value),\n AveRooms and "
      "AveBedrm, AveRooms and target, MedInc and AveRooms. There is strong negative correlation between HouseAge and MedInc, \n HouseAge "
      "and AveRoom, HouseAge and Population, Latitude and Longitude.")

warnings.filterwarnings("ignore")

l = df.columns.values.tolist()
fig = plt.figure(figsize=[9, 9])
ax = fig.add_subplot(111)
ax.set_xticklabels([''] + l)
ax.set_yticklabels([''] + l)
cax = ax.matshow(correl_np)
plt.show()

warnings.simplefilter('always')

print("4.2","_"*50)

hollywood = df.loc[(33 <= df['Latitude']) & (df['Latitude'] <= 36) & (-116 >= df['Longitude']) &  (df['Longitude'] >= -120)]

plt.scatter(x=hollywood['Longitude'], y=hollywood['Latitude'])
plt.show()

df['isOld'] = (df['HouseAge']>25).astype(int)
df['greaterThan3'] = (df['AveBedrms']>3).astype(int)
df['hollywood'] = (df.index.isin(hollywood.index)).astype(int)

df['isOld']=df['isOld'].astype(str)
df['greaterThan3']=df['greaterThan3'].astype(str)
df['hollywood']=df['hollywood'].astype(str)

df['HOL'] = df['isOld'] +"_" + df['greaterThan3'] +"_" + df['hollywood']

print(df['HOL'].value_counts())

df = df.drop(columns=['isOld', 'greaterThan3', 'hollywood' ])

df = pd.get_dummies(df, columns=['HOL'])

print("4.3","_"*50)

y = df[['target']].values

X = df.drop(columns='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("4.4","_"*50)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

predictions1 = model.predict(X_train)

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

mae1 = metrics.mean_absolute_error(y_train, predictions1)
mse1 = metrics.mean_squared_error(y_train, predictions1)
rmse1 = np.sqrt(metrics.mean_squared_error(y_train, predictions1))

print(f"MAE for test value equals {round(mae, 5)}. In other words, the average difference between predictions and "
      f"test is {round(mae, 2)}.")
print(f"MSE for test value equals {round(mse, 5)}")
print(f"RMSE for test value equals {round(rmse, 5)}")

print(f"MAE for train value equals {round(mae1, 5)}")
print(f"MSE for train value equals {round(mse1, 5)}")
print(f"RMSE for train value equals {round(rmse1, 5)}")

print("It seems like the model overfits. Cross-validation would be helpful.")

print("5.1","_"*50)

df = pd.DataFrame(house_data.data, columns=house_data.feature_names)

target = pd.Series(house_data.target)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20)

print("5.2", "_"*50)

model = Lasso(alpha=1.0)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("5.3", "_"*50)

mae2 = metrics.mean_absolute_error(y_test, predictions)

print(f"MAE for test value equals {round(mae2, 5)}")

print("5.4", "_"*50)

print("Model coefficients:", model.coef_)

print('Features with coefficients shrank to zero: {}'.format(
      np.sum(model.coef_ == 0)))

removed_features = X_train.columns[(model.coef_ == 0).ravel().tolist()]

print(removed_features.values)

print("5.5", "_"*50)

X_test = X_test.drop(columns=removed_features.values)
X_train = X_train.drop(columns=removed_features.values)

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
print("Model coefficients:", model.coef_)
predictions = model.predict(X_test)

mae3 = metrics.mean_absolute_error(y_test, predictions)

print(f"MAE for test value equals {round(mae3, 5)}")

print("I guess it's logical that MAE didn't change because shrank coefficients hadn't be considered by the first model "
      "to begin with. So, their removal changed nothing")

