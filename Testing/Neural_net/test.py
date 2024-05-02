import spacy
import csv
import pandas as pd
import numpy as np

def get_validation_set_and_process_data(path):
  validation_set = pd.read_csv(path)

  num_examples = validation_set.shape[0]
  label = np.zeros((1, num_examples))

  # Shape of each 1 by 7 array
  shape_of_each_example = (1, 24)
  # Create the NumPy array filled with zeros
  matrix = np.zeros((num_examples, shape_of_each_example[1]))

  for index, row in validation_set.iterrows():
    if ( str(row['label_boolean']) == "True"):
      label[0,index] = 1
    if ( str(row['label_boolean']) == "False"):
      label[0,index] = 0

    boolean_array = check_words_presence(row['text'])
    numpy_array = np.array(boolean_array, dtype=int)
    numpy_array = np.insert(numpy_array, 0, row['metaphorID'], axis=0)
    parts_of_speech_freq = get_part_0f_speech_frequencies(row['text'])
    target_encoding_and_pos_encoding = np.concatenate((numpy_array, parts_of_speech_freq))
    marker_freq_array = get_marker_counts(row['text'])
    final_feature = np.concatenate((target_encoding_and_pos_encoding, marker_freq_array))
    matrix[index] = final_feature

    # print(num_examples)
    # return matrix, label


  return matrix, label


nlp_processor = spacy.load("en_core_web_sm")



#encodes the metaphorical target with 1 and the rest of the words if not found with zero
def check_words_presence(sentence):
    words_to_check = ["road", "candle", "light", "spice", "ride", "train", "boat"]
    sentence = sentence.lower()  # Convert to lowercase for case-insensitive matching
    return [word in sentence for word in words_to_check]

def get_part_0f_speech_frequencies(doc):
    doc = nlp_processor(doc)
    pos_frequencies = np.zeros(8)

    # Define a mapping of part of speech labels to indices
    pos_mapping = {"NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "PRON": 4, "ADP": 5, "CONJ": 6, "NUM": 7}

    # Count the frequencies of each part of speech
    for token in doc:
      pos = token.pos_
      if pos in pos_mapping:
          pos_index = pos_mapping[pos]
          pos_frequencies[pos_index] += 1

    return pos_frequencies
    # Print the one-dimensional array of part of speech frequencies
    print("Part of Speech Frequencies:", pos_frequencies)

def get_marker_counts(sentence):
  metaphor_marker = ["!", "as", "believe", "just", "like", "could", "really", "would"]



  # Process the sentence using spaCy
  doc = nlp_processor(sentence)

  # Initialize a list to store the counts
  counts = [0] * len(metaphor_marker)

  # Loop through the tokens in the processed sentence
  for token in doc:
      # Check if the token text is in the list of words to count
      if token.text in metaphor_marker:
          # Find the index of the token text in the list and increment the corresponding count
          index = metaphor_marker.index(token.text)
          counts[index] += 1

  # Convert the list of counts to a NumPy array
  counts_array = np.array(counts)

  return counts_array




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_predictions(X, Y):
  w1 = np.load('weights/w1_c.npy', allow_pickle=True)
  b1 = np.load('weights/b1_c.npy', allow_pickle=True)
  w2 = np.load('weights/w2_c.npy', allow_pickle=True)
  b2 = np.load('weights/b2_c.npy', allow_pickle=True)

  z1 = np.dot(w1, X.T)
  a1 = sigmoid(z1)
  z2 = np.dot(w2,a1)
  a2 = sigmoid(z2)

  # Threshold value
  threshold = 0.4
  # Convert values below the threshold to 0 and values above the threshold to 1
  predicted_label = (a2 > threshold).astype(int)

  return predicted_label



  def calculate_accuracy(y_true, y_pred):
    # Ensure the arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    # # Compare the two arrays element-wise and count the number of matches
    # correct_predictions = np.sum(np.isclose(y_true, y_pred, rtol=1e-05, atol=1e-08))

    # Calculate the accuracy as the percentage of correct predictions
    accuracy = (y_true == y_pred).mean() * 100.0
    return accuracy


def calculate_confusion_matrix(true_labels, predicted_labels):
    # Ensure the inputs are NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # True Positive (TP)
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))

    # False Positive (FP)
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))

    # True Negative (TN)
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))

    # False Negative (FN)
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


    return precision, recall, f1_score


path = 'dataset/validation_set.csv'
# path = '/content/drive/MyDrive/stat_ml_data/training_set.csv'
processed_data, label = get_validation_set_and_process_data(path)

y_pred = get_predictions(processed_data,label)
# print(processed_data.shape, label.shape)
# processed_data, label = get_features_from_validation_set(validation_set)
# print(validation_set.head)
# print(label)
# print(processed_data)


# print(label.shape)
# print(y_pred.shape)
# print(label)
# print(y_pred)
accuracy = calculate_accuracy(label, y_pred)
precision, recall, f1_score = calculate_confusion_matrix(label, y_pred)
print(f"Precision: {precision:.2f}%")
print(f"recall: {recall:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"f1 score: {f1_score:.2f}%")




  