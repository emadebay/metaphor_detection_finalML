import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

# Load the dataset
file_path = 'train.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Split the dataset into training and validation sets
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the split data into separate files
train_data.to_csv('train_split.csv', index=False)
validation_data.to_csv('validation_split.csv', index=False)

# Initialize a TF-IDF Vectorizer and Random Forest model within a pipeline
pipeline_rf = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42 ,n_estimators= 250))

# Train the Random Forest model on the training data
pipeline_rf.fit(train_data['text'], train_data['label_boolean'])
# Get the number of trees in the Random Forest
num_trees = pipeline_rf.named_steps['randomforestclassifier'].n_estimators
print(f'Number of trees in the Random Forest: {num_trees}')

# Make predictions on the validation set
validation_predictions_rf = pipeline_rf.predict(validation_data['text'])

# Calculate performance metrics for Random Forest model
accuracy_rf = accuracy_score(validation_data['label_boolean'], validation_predictions_rf)
precision_rf = precision_score(validation_data['label_boolean'], validation_predictions_rf, average='macro')
recall_rf = recall_score(validation_data['label_boolean'], validation_predictions_rf, average='macro')
f1_rf = f1_score(validation_data['label_boolean'], validation_predictions_rf, average='macro')

# Save the predictions to a CSV file
validation_data['predictions_rf'] = validation_predictions_rf
validation_data.to_csv('predictions_random_forest.csv', index=False)

print(f'Accuracy: {accuracy_rf}')
print(f'Precision: {precision_rf}')
print(f'Recall: {recall_rf}')
print(f'F1 Score: {f1_rf}')


import matplotlib.pyplot as plt

# Extract feature importances from the Random Forest model
importances = pipeline_rf.named_steps['randomforestclassifier'].feature_importances_

# Extract feature names from the TF-IDF Vectorizer
feature_names = pipeline_rf.named_steps['tfidfvectorizer'].get_feature_names_out()

# Sort the features by their importance
sorted_indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.title("Top 10 Feature Importances in the Random Forest Model")
plt.bar(range(10), importances[sorted_indices[:10]], align='center')
plt.xticks(range(10), feature_names[sorted_indices[:10]], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Select a single tree from the Random Forest
single_tree = pipeline_rf.named_steps['randomforestclassifier'].estimators_[0]

# Convert the feature names from a NumPy array to a list
feature_names_list = feature_names.tolist()

plt.figure(figsize=(20, 10))
plot_tree(single_tree, feature_names=feature_names_list, max_depth=5, filled=True)
plt.show()

# Assuming 'single_tree' is your trained DecisionTreeClassifier object

# Root split feature index and threshold
root_feature_index = single_tree.tree_.feature[0]
root_threshold = single_tree.tree_.threshold[0]

# Number of branches (non-leaf nodes)
# A non-leaf node has a feature index that is not equal to _tree.TREE_UNDEFINED
# which is defined as -2 in scikit-learn
num_branches = sum(single_tree.tree_.feature != -2) - 1  # Subtracting one because root is not a branch

# Depth of the tree
depth_of_tree = single_tree.get_depth()

# Number of leaf nodes
num_leaf_nodes = single_tree.get_n_leaves()

root_feature_name = feature_names_list[root_feature_index]
print(f'Root split: Feature {root_feature_name} at threshold {root_threshold}')
print(f'Number of branches: {num_branches}')
print(f'Depth of the tree: {depth_of_tree}')
print(f'Number of leaf nodes: {num_leaf_nodes}')
