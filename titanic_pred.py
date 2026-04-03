import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


titanic_df = pd.read_csv("C:/Users/GMI/Downloads/Titanic-Dataset.csv")
print(titanic_df.head())
print("--------------------------------------------------------------------")
print("DataFrame shape: ",titanic_df.shape)
print("--------------------------------------------------------------------")
print("== Missing Values ==")
print(titanic_df.isnull().sum())
print('== All missing values ==')
print(titanic_df.isnull().sum().sum())
print("--------------------------------------------------------------------")
print("== Descriptive Statistics ==")
print(titanic_df.describe())
print(titanic_df.info())
print("--------------------------------------------------------------------")
print("== For training and testing sets ==")
columns_names = titanic_df.columns.tolist()
print(f"Features: {columns_names[:1]+columns_names[2:]} ", )
print("Target: 'Survived' ")
print("--------------------------------------------------------------------")

plt.hist(titanic_df["Survived"], bins=2, edgecolor="black",color="red")
plt.title("Histogram of Survived")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

titanic_clean1 = titanic_df.copy()

np.random.seed(42)
print(titanic_clean1.isna().sum())
#print(titanic_clean1[["Pclass", "Cabin"]].head(20))
test = titanic_clean1[(titanic_clean1["Pclass"] == 1) & (titanic_clean1["Cabin"].isna())]
#grpd = titanic_clean1["Cabin"].isna().groupby(titanic_clean1["Pclass"]).sum()
#print(grpd)

titanic_clean2 = titanic_clean1.drop("Cabin", axis=1)

#print(dropped_cabins["Sex"].value_counts())

# Clean and replace in one go
titanic_clean2['Sex'] = (titanic_clean2['Sex'].astype(str).str.strip().str.lower().map({'female': 0, 'male': 1})
                        .fillna(-1)
                        .astype(int))
#print(titanic_clean2["Sex"].head(10))

titanic_clean2["Age"] = titanic_clean2["Age"].fillna(titanic_clean2["Age"].median()).astype(int)

titanic_clean2["Embarked"] = titanic_clean2["Embarked"].fillna('C')

print(titanic_clean2["Fare"].describe())
med_fare = titanic_clean2["Fare"].median()

def fare_category(fare):
    if fare > med_fare:
        return 1 #for expensive
    else:
        return 0 #for cheap

titanic_clean2["Fare"] = titanic_clean2["Fare"].apply(fare_category)
#print(titanic_clean2["Fare"].head(10))
titanic_clean2["FamilySize"] = titanic_clean2["SibSp"] + titanic_clean2["Parch"] + 1
#print(titanic_clean2["FamilySize"].value_counts())

print(titanic_clean2.dtypes)
no_name_titanic = titanic_clean2.drop("Name", axis=1)
no_name_titanic["Ticket"].fillna(0)
print(no_name_titanic.isna().any())
print(no_name_titanic["Ticket"].values)


no_name_titanic["TicketPrefix"] = no_name_titanic["Ticket"].str.extract(r"^([A-Za-z/.]+)")
no_name_titanic["TicketPrefix"] = no_name_titanic["TicketPrefix"].fillna("NONE")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
no_name_titanic["TicketPrefix_encoded"] = encoder.fit_transform(no_name_titanic["TicketPrefix"])
no_name_titanic.drop("TicketPrefix", axis=1, inplace=True)
no_name_titanic.drop("Ticket", axis=1, inplace=True)
no_name_titanic["Embarked_encoded"] = encoder.fit_transform(no_name_titanic["Embarked"])
no_name_titanic.drop("Embarked", axis=1, inplace=True)


print(no_name_titanic.head(15))
print(no_name_titanic.dtypes)

titanic3 = no_name_titanic.drop("SibSp", axis=1)
titanic4 = no_name_titanic.drop("Parch", axis=1)

feature_columns = [ "PassengerId","Pclass","Sex","Age","Fare","FamilySize","TicketPrefix_encoded","Embarked_encoded" ]

X = titanic3[feature_columns]
y = titanic3["Survived"]

print("Features shape: ", X.shape)
print("Target shape: ", y.shape)
print("\nFeature columns:")
for i, col in enumerate(X.columns.tolist(), 1):
    print(f"  {i}. {col}")

corr_data = titanic3[feature_columns+["Survived"]]
corr_mat = corr_data.corr()

sns.heatmap(corr_mat, annot=True, fmt='.2f',cmap="coolwarm", center=0, linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Survived:")
print(corr_mat['Survived'].sort_values(ascending=False))

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split complete!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled!")
print("\nBefore scaling (first sample):")
print(X_train.iloc[0].values)
print("\nAfter scaling (first sample):")
print(X_train_scaled[0])

logreg.fit(X_train_scaled, y_train)
y_pred_train = logreg.predict(X_train_scaled)
y_pred_test = logreg.predict(X_test_scaled)

y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]
from sklearn.metrics import*

print(" -- Logistic Regression Performance -- ")
print(" - On Training Set - ")
print("Accuracy: ", accuracy_score(y_train, y_pred_train))
print("Precision: ", precision_score(y_train, y_pred_train))
print("Recall: ", recall_score(y_train, y_pred_train))
print("F1: ", f1_score(y_train, y_pred_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))

print(" - On Test Set - ")
print("Accuracy: ", accuracy_score(y_test, y_pred_test))
print("Precision: ", precision_score(y_test, y_pred_test))
print("Recall: ", recall_score(y_test, y_pred_test))
print("F1: ", f1_score(y_test, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_test))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train_scaled, y_train)
    train_accuracies[neighbor] = knn.score(X_train_scaled, y_train)
    test_accuracies[neighbor] = knn.score(X_test_scaled, y_test)

plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

print("-- KNN Performance -- ")
print("peaked at almost 80% accuracy when k=12")

print("GET READY MY MODEL, HERE COMES SOME UNSEEN DATA !!")


# Define predictor function using individual features
def predict_survival(pclass, sex, age, fare, family_size, ticket_prefix_encoded, embarked_encoded):
    df = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "FamilySize": family_size,
        "TicketPrefix_encoded": ticket_prefix_encoded,
        "Embarked_encoded": embarked_encoded
    }])
    df_scaled = scaler.transform(df)

    pred_log = logreg.predict(df_scaled)[0]
    prob_log = float(logreg.predict_proba(df_scaled)[:, 1][0])
    prob_log = round(prob_log, 3)

    pred_knn = knn.predict(df_scaled)[0]

    return {
        "Logistic Regression": "Yes" if pred_log == 1 else "No",
        "LR Probability": prob_log,
        "KNN": "Yes" if pred_knn == 1 else "No"
    }


# -------------------------
# Example calls
# -------------------------

# Example 1
example1 = predict_survival(pclass=3, sex=1, age=29, fare=0, family_size=1, ticket_prefix_encoded=0, embarked_encoded=2)
print("Example 1: 29-year-old male, 3rd class, predicted:", example1)

# Example 2
example2 = predict_survival(pclass=1, sex=0, age=35, fare=1, family_size=2, ticket_prefix_encoded=5, embarked_encoded=0)
print("Example 2: 35-year-old female, 1st class, predicted:", example2)

# Example 3
example3 = predict_survival(pclass=2, sex=1, age=22, fare=0, family_size=1, ticket_prefix_encoded=3, embarked_encoded=1)
print("Example 3: 22-year-old male, 2nd class, predicted:", example3)
