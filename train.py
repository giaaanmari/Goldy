import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, GaussianNoise
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import random
words=[]
classes = []
documents = []
ignore_words = ['?', '!', '.']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)
stop_words = set(stopwords.words('english'))
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words or w not in stop_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(tuple(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
split_set= int(0.8 * len(training))

train_set = training[:split_set]
test_set = training[split_set:]
# create train and test lists. X - patterns, Y - intents
train_x = list(train_set[:,0])
train_y = list(train_set[:,1])

test_x = list(test_set[:,0])
test_y = list(test_set[:,1])
print("Training data created")
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train, val in kfold.split(train_x, train_y):
    hist = model.fit(train_x, train_y, epochs=64, batch_size=10, verbose=2)
    for i in val:
        if i < len(test_x):
            scores = model.evaluate(test_x[i:i+1], test_y[i:i+1], verbose=0)
            print(f"Validation loss: {scores[0]} / Validation accuracy: {scores[1]}")
model.save('model.h5', hist)


# MODEL ACCURACY / MODEL LOSS
print("model created")
