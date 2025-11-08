import pickle
# Random Forest is an ensemble machine learning model that builds many decision trees
# and averages their results to make predictions more robust and less prone to overfitting
from sklearn.ensemble import RandomForestClassifier
# `train_test_split` is used to evaluate how well the model generalizes to unseen data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Loads the preprocessed dataset that was saved earlier using pickle
data_file = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_file['data'])
labels = np.asarray(data_file['labels'])
# Splits the dataset into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
# Trains the model using the training data and corresponding labels
model.fit(x_train, y_train)
# Predicts the labels for the test data (unseen samples)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly')
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()