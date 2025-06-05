import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df=pd.read_csv("e-learning.csv")
print(df.head())
print(df.shape)
print(df.columns.to_list())
df.replace(['#DIV/0!', 'NaN', 'NULL', ''], np.nan, inplace=True)
df['Target'].value_counts(normalize=True)

print(df.groupby('Gender')['Target'].value_counts(normalize=True))
print(df.groupby('Scholarship holder')['Target'].value_counts(normalize=True))
print(df.groupby('Debtor')['Target'].value_counts(normalize=True))

#EDA
sns.countplot(x='Target',data=df)
plt.title('Student Dropout Status (0 = Enrolled, 1 = Graduate, 2 = Dropout)')

plt.figure(figsize=(20,20))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.countplot(x='Gender', hue='Target', data=df)
plt.title('Dropout by Gender')

sns.countplot(x='Scholarship holder', hue='Target', data=df)
plt.title('Dropout by Scholarship Status')
plt.show()

df['Engagement_Level'] = df['Total Evaluations'].apply(lambda x: 'Active' if x > 4 else 'Inactive')
sns.countplot(data=df, x='Engagement_Level', palette='Set2')
plt.title('Active vs Inactive Learners')
plt.show()

sns.countplot(data=df, x='Engagement_Level', hue='Target', palette='Set1')
plt.title('Dropout Rate by Engagement Level')
plt.legend(title='Dropout (1=Yes, 0=No)')
plt.show()

sns.boxplot(data=df, x='Engagement_Level', y='Average Grade', palette='Pastel1')
plt.title('Average Grade by Engagement Level')
plt.show()


#regression model to predict the likelihood of a student dropping out (1 = dropout, 0 = not dropout)

df['dropout_flag'] = df['Target'].apply(lambda x: 1 if x == 2 else 0)
features = [
    'Age at enrollment', 'Gender', 'Scholarship holder', 'Debtor',
    'Tuition fees up to date', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Total Evaluations', 'Average Grade'
]

df_encoded = df.copy()
le = LabelEncoder()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])  # Example: M = 1, F = 0

X = df_encoded[features]
y = df_encoded['dropout_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Predict using the trained model
predictions = model.predict(X_imputed)
df_encoded['predicted_dropout_risk'] = predictions
# Add predictions to the DataFrame
df_encoded['predicted_dropout_risk'] = predictions
df_encoded['high_risk_flag'] = df_encoded['predicted_dropout_risk'].apply(lambda x: 1 if x > 0.6 else 0)

#Histogram of Dropout Risk
plt.figure(figsize=(8,5))
sns.histplot(df_encoded['predicted_dropout_risk'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Predicted Dropout Risk')
plt.xlabel('Predicted Dropout Risk Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()

#Highlight High-Risk Students (Risk > 0.6)
high_risk = df_encoded[df_encoded['predicted_dropout_risk'] > 0.6]
print(f"Number of High-Risk Students: {len(high_risk)}")

# Preview top high-risk students
high_risk[['Age at enrollment', 'Gender', 'Course', 'Average Grade', 'predicted_dropout_risk']].sort_values(
    by='predicted_dropout_risk', ascending=False).head(10)

#Boxplot: Dropout Risk by Gender
plt.figure(figsize=(6,4))
sns.boxplot(data=df_encoded, x='Gender', y='predicted_dropout_risk')
plt.title('Dropout Risk by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Predicted Dropout Risk')
plt.show()

df_final = df_encoded.rename(columns={
    'Curricular units 1st sem (evaluations)': 'Evaluations_1st_Sem',
    'Curricular units 2nd sem (evaluations)': 'Evaluations_2nd_Sem',
    'Curricular units 1st sem (grade)': 'Grade_1st_Sem',
    'Curricular units 2nd sem (grade)': 'Grade_2nd_Sem',
    'Age at enrollment': 'Age',
    'Total Evaluations': 'Total_Evaluations',
    'Average Grade': 'Average_Grade',
    'predicted_dropout_risk': 'Predicted_Dropout_Risk',
})

cols_order = [
    'Age', 'Gender', 'Course', 'Scholarship holder', 'Debtor',
    'Total_Evaluations', 'Average_Grade',
    'Predicted_Dropout_Risk', 'High_Risk_Flag',
    'Target'  # ground truth (if retained)
]

# Include only those columns that exist
cols_order = [col for col in cols_order if col in df_final.columns]

df_final = df_final[cols_order + [col for col in df_final.columns if col not in cols_order]]
df_final.to_csv("final_e-learning_analysis.csv", index=False)
