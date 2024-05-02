import spacy
import csv
import pandas as pd
import numpy as np

def get_training_set_and_process_data(path):
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


def initialize_weights_and_bias(X, Y):

    X = X.T
    shape_of_x = X.shape
    num_of_hidden_unit = 1000
    # w1 = np.random.randn(num_of_hidden_unit, shape_of_x[0]) * 0.001
    # b1 = np.random.randn(num_of_hidden_unit, 1) * 0.001
    # w2 = np.random.randn(1, num_of_hidden_unit)* 0.001
    # b2 = np.random.randn(1, 1)* 0.001
    w1 = np.load('weights/w1_c.npy', allow_pickle=True)
    b1 = np.load('weights/b1_c.npy', allow_pickle=True)
    w2 = np.load('weightsw/2_c.npy', allow_pickle=True)
    b2 = np.load('weights/b2_c.npy', allow_pickle=True)

    print(w1.shape)
    print(b1.shape)
    print(w2.shape)
    print(b2.shape)



    dynamic_dic = {}

    dynamic_dic["w1"] = w1
    dynamic_dic["b1"] = b1
    dynamic_dic["w2"] = w2
    dynamic_dic["b2"] = b2
    dynamic_dic["X"] = X
    dynamic_dic["Y"] = Y
    dynamic_dic["shape_of_x"] = X.shape
    return dynamic_dic


def forward_propagation_and_backward_propgation(weights_and_biases):
  w1 = weights_and_biases["w1"]
  b1 = weights_and_biases["b1"]
  w2 = weights_and_biases["w2"]
  b2 = weights_and_biases["b2"]
  X = weights_and_biases["X"]
  Y = weights_and_biases["Y"]

  training_data_shape = weights_and_biases["shape_of_x"]
  len_of_training_data = training_data_shape[1]

  z1 = np.dot(w1, X)
  weights_and_biases["z1"] = z1
  a1 = sigmoid(z1)
  weights_and_biases["a1"] = a1
  z2 = np.dot(w2,a1)
  weights_and_biases["z2"] = z2
  a2 = sigmoid(z2)
  weights_and_biases["a2"] = a2

  dy_da2 = -( (Y/a2) - (1-Y)/(1-a2) )
  da2_dz2 = a2 * (1 - a2)
  dz2_da2 = a1
  dz2_da1 = w2
  dz2_db1 = 1
  da1_dz1 = a1 * (1 - a1)
  dz1_dw1 = X
  dz1_db1 = 1

  dz2 = dy_da2 * da2_dz2
  dw2 = np.dot(dz2, a1.T) / len_of_training_data
  db2 = np.sum(dz2, axis=1, keepdims=True) / len_of_training_data

  da1 = np.dot(w2.T, dz2)
  dz1 = da1 * da1_dz1
  dw1 = np.dot(dz1, X.T) / len_of_training_data
  db1 = np.sum(dz1, axis=1, keepdims=True) / len_of_training_data

  loss = - (Y * np.log(a2) + (1 - Y) * np.log(1 - a2))

  w2 = w2 - (0.01 * dw2)
  b2 = b2 - (0.01 * db2)  # Corrected from b1
  w1 = w1 - (0.01 * dw1)
  b1 = b1 - (0.01 * db1)

  weights_and_biases["w1"] = w1
  weights_and_biases["b1"] = b1
  weights_and_biases["w2"] = w2
  weights_and_biases["b2"] = b2



  print(np.sum(loss))

  return w1,b1,w2,b2



path = 'dataset/training_set.csv'
processed_data, label = get_training_set_and_process_data(path)


# print(processed_data.shape, label.shape)
# print(processed_data[0])
ws_and_bs = initialize_weights_and_bias(processed_data, label)
epoch = 40001
for i in range (epoch):
  w1,b1,w2,b2 = forward_propagation_and_backward_propgation(ws_and_bs)
  if i % 100 == 0:
    np.save('weights/w1_c.npy', w1)
    np.save('weights/b1_c.npy', b1)
    np.save('weightsw/2_c.npy', w2)
    np.save('weights/b2_c.npy', b2)