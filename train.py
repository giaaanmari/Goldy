import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.metrics import Precision, Recall
import random
import re
words=[]
tags = []
context = []
documents = []
ignore_words = ['.','?','!',':',';','<','>',',','\'']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:   
    for pattern in intent['patterns']:
        #tokenize each word
        w = re.sub(r'[^\w\s]', '', pattern)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])
        for unique_context in intent['context']:
            c = nltk.word_tokenize(unique_context)
            context.extend(c)


# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words and w not in stopwords.words('english')]
words = sorted(list(set(words)))
# sort classes
tags = [list(c) for c in set(tuple(i) for i in tags)]
tags = sorted(tags)

context = sorted(list(set(context)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(tags), "tags", tags)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

print(len(context), "unique contexts:", context)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))
pickle.dump(context,open('context.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(tags)
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
    output_row[tags.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
# create train and test lists. X - patterns, Y - intents
#train_x, val_x, train_y, val_y = train_test_split(list(training[:,0]), list(training[:,1]), test_size=0.2, random_state=42)
print("Training data created")
# Create model - 3 layers. First layer 128 neurons, second layer Dropout Layer with 0.1 value and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.11, decay=1e-6, momentum=0.75, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', Precision(), Recall()])
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=64, batch_size=164, validation_split=0.2, verbose=2)
model.save('model.h5', hist)
#score = model.evaluate(val_x, val_y, batch_size=64)
#print('Validation Loss: {:.4f}'.format(score[0]))
#print('Validation Accuracy: {:.4f}'.format(score[1]))


####### PLOTTING

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['precision'])
plt.plot(hist.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['precision', 'val_precision'], loc='upper left')
plt.show()

plt.plot(hist.history['recall'])
plt.plot(hist.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['recall', 'val_recall'], loc='upper left')
plt.show()



print(model.summary())