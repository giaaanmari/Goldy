    #### IMPORT LIBRARIES

import os
import requests
import pathlib
from difflib import get_close_matches
import google.oauth2.credentials
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import pickle
import enchant
import string
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import re

stop_words = set(stopwords.words('english'))
nltk.download('popular')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
context = pickle.load(open('context.pkl','rb'))
previous_context = []
current_context = None
similarity_threshold = 0.8

dictionary = enchant.Dict("en_US")
for word in words:
    dictionary.add(word)

default_responses = ["I'm sorry, I didn't quite understand what you said. Can you please try asking me again in a different way?",
                     "I'm sorry, I don't have the answer to that question right now. But don't worry, I'll keep learning and hopefully, I'll be able to help you with your question soon.",
                     "Hmm, I'm not quite sure what you're asking. Can you please give me more information or context about your question?",
                     "I'm sorry, but I'm not sure what you're trying to say. Can you please provide me with more information or a specific question so I can better assist you?",
                     "I'm having trouble understanding what you're trying to communicate. Can you please provide me with more context or a specific topic so I can better assist you?"]

#### PRE-PROCCESSING
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    tokens = [w for w in (nltk.word_tokenize(sentence))]

    # Define the set of stopwords excluding interrogative pronouns
    custom_stopwords = set(stopwords.words('english')) - {'who', 'what', 'when', 'where', 'why', 'which', 'whom', 'how'}

    # stem each word - create short form for word
    clean_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in custom_stopwords]

    return clean_tokens

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.8
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": tags[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json, msg):
    global default_responses
    global context
    global current_context
    global previous_context
    if len(ints) > 0:
        tag = ints[0]['intent']
    else:
        tag = None

    list_of_intents = intents_json['intents']

    input_checked = check_input(msg)
    if input_checked:
        print("input checked")
        return input_checked    
    
    for i in list_of_intents:
        if tag == i['tag']:
            possible_responses = i['responses']
            if current_context == None:
                current_context = i['context']
            else:
                previous_context = current_context
                current_context = i['context']
            return random.choice(possible_responses)
        
    previous_context = current_context
    current_context = None

    cleaned_tokens = clean_tokens(nltk.word_tokenize(msg))
    if len(cleaned_tokens) == 1:
        word = cleaned_tokens[0]
        if get_definition(word) != None:
            return get_definition(word)
    
    return random.choice(default_responses)

def clean_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

def check_input(msg):
    pos = ['noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', 'interjection']
    ask = ['define', 'definition', 'explain', 'meaning', 'means', 'mean']
    
    if checkKey(msg):
        return False
    
    for word in msg.split():
        split_msg = msg.split()
        if word in pos:
            split_msg.remove(word)
            new_msg = " ".join(split_msg)
            clean_new_msg = clean_tokens(nltk.word_tokenize(new_msg))
            if clean_new_msg:
                return get_POS(clean_new_msg, word)
        if word in ask:
            split_msg.remove(word)
            new_msg = " ".join(split_msg)
            clean_new_msg = clean_tokens(nltk.word_tokenize(new_msg))
            if clean_new_msg:
                word = clean_new_msg[0]
                return get_definition(word)
        if word == "synonyms" or word == "synonym":
            split_msg.remove(word)
            new_msg = " ".join(split_msg)
            clean_new_msg = clean_tokens(nltk.word_tokenize(new_msg))
            if clean_new_msg:
                word = clean_new_msg[0]
                synonyms, antonyms = get_synonyms_antonyms(word)
                synonyms = list(set(synonyms))
                syns = "\n".join(synonyms) if synonyms else None
                if syns == None:
                    res = f"I'm sorry, but I couldn't find any other words that have the same meaning as {word}."
                else:
                    res = f"Here are the synonyms of {word}:\n{syns}"
                return res
        if word == "antonyms" or word == "antonym":
            split_msg.remove(word)
            new_msg = " ".join(split_msg)
            clean_new_msg = clean_tokens(nltk.word_tokenize(new_msg))
            if clean_new_msg:
                word = clean_new_msg[0]
                synonyms, antonyms = get_synonyms_antonyms(word)
                antonyms = list(set(antonyms))
                ants = "\n".join(antonyms) if antonyms else None
                if ants == None:
                    res = f"I'm sorry, but I couldn't find any opposite words for {word}."
                else:
                    res = f"Here are the antonyms of {word}:\n{ants}"
                return res

    return False


# code if the user wants to know the POS of a word
def get_POS(cleaned_str, pos):
    vowels = set("aeiouAEIOU")
    if pos[0] in vowels:
        article = "an"
    else:
        article = "a"

    word = cleaned_str[0]
    synsets = wordnet.synsets(word)
    pos_list = [synset.pos() for synset in synsets]
    print(pos_list)
    if pos[0] in pos_list:
        definition = get_definition_and_pos(word, pos, article, True)
        return definition
    else: 
        definition = get_definition_and_pos(word, pos, article, False)
        return f"No, the word {word} is not {article} {pos}. {definition}"
        

def get_definition_and_pos(word, input_pos, article, isFound): # using WordNet
    pos_tag = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r", "preposition": "p", "conjunction": "c", "interjection": "u"}.get(input_pos)
    synsets = wordnet.synsets(word) # get definition
    if synsets:
        pos = synsets[0].pos()
        pos_word = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "p": "preposition", "c": "conjunction", "u": "interjection"}.get(pos)
        definition = synsets[0].definition()
        if pos_word == input_pos:
            ans = f"Yes, {word} is {article} {pos_word}. "
            synsets = wordnet.synsets(word, pos=pos)
            definitions = [synset.definition() for synset in synsets]
            # if definitions:
            ans += f"Here are the definitions of {word} as {article} {pos_word}:\n"
            for i, definition in enumerate(definitions):
                ans += f"- {definition}\n"
        elif isFound:
            ans = f"Yes, {word} can also be used as {article} {input_pos}. "
            synsets = wordnet.synsets(word, pos=pos_tag)
            # if synsets:
            definitions = [synset.definition() for synset in synsets]
            # if definitions:
            ans += f"When used as {article} {input_pos}, it means:\n"
            for i, definition in enumerate(definitions):
                ans += f"- {definition}\n"
        else:
            ans =  f"The part of speech of {word} is {pos_word}. It is defined as {definition}."
        return ans
    return None

def get_definition(word):
    print("define: ", word)
    synsets = wordnet.synsets(word)
    if synsets:
        definition = synsets[0].definition()
        return f"The word {word} means " + definition + "."
    return None

def get_synonyms_antonyms(word):
    synonyms = []
    antonyms = []

    synsets = wordnet.synsets(word)

    for synset in synsets:
        for lemma in synset.lemmas():
            # synonyms.append(lemma.name().lower().replace('_', ' '))
            synonym = lemma.name().lower().replace('_', ' ')  # Convert synonym to lowercase and remove underscores
            synonyms.append(synonym)
            if lemma.antonyms():
                # antonyms.append(lemma.antonyms()[0].name().lower().replace('_', ' '))
                antonym = lemma.antonyms()[0].name().lower().replace('_', ' ')  # Convert antonym to lowercase and remove underscores
                antonyms.append(antonym)
                antonyms += [ant.replace('_', ' ') for ant in lemma.antonyms()[1:]]

    return synonyms, antonyms

def checkKey(str):
    print("key checked")
    sentence_words = clean_tokens(nltk.word_tokenize(str))
    key_example = ["example", "examples", "instance", "instances", "illustrate", "sample", "samples"]
    key_type = ["type", "types", "kind", "kinds"]
    user_input = set(sentence_words)
    example_keyword = set(key_example)
    type_keyword = set(key_type)
    # check if the input is a keyword
    isExample = user_input.intersection(example_keyword)
    isType = user_input.intersection(type_keyword)
    str = " ".join(user_input)

    tempKey = ""
    key = ""
    g = False
    if isExample:
        key = "example"
        tempKey = "".join(isExample)
        g = True
    if isType:
        key = "type"
        tempKey = "".join(isType)
        g = True
    
    str = [element for element in user_input if element != tempKey]

    if g and current_context != None and len(str) == 0:
        context = " ".join(current_context)
        msg = key + ' ' + context
        return msg
    if g and current_context == None and len(str) != 0:
        str = " ".join(str)
        msg = key + ' ' + str
        return msg
    
    return False

def spell(match):
    word = match.group(0)
    return dictionary.suggest(word)[0] if not dictionary.check(word) and dictionary.suggest(word) else word
    
flag = False
text = []

def chatbot_response(input_msg):
    global text
    global flag
    global context
    global default_responses

    res = ["Can you please clarify what you're asking or provide more information so I can better assist you?",
           "Okay. Could you please clarify your question so I can assist you better?",
           "Okay. Could you please rephrase your question or provide more details so I can better understand how to help you?"]

    input_msg = input_msg.lower()

    correct_msg = re.sub(r"\w+", spell, input_msg) # correct the spelling of the input_msg

    msg = checkKey(correct_msg)
    if msg:
        correct_msg = msg
        input_msg = correct_msg 
        print("new message: ", correct_msg)

    if flag:
        output_word=[correct_msg for correct_msg in text]
        output_txt=" ".join(output_word)
        if correct_msg == "yes" or correct_msg in output_txt:
            ints = predict_class(output_txt, model)
            res = getResponse(ints, intents, output_txt)
        elif correct_msg == "no":
            res = random.choice(res)
        elif correct_msg == input_msg:
            ints = predict_class(correct_msg, model)
            res = getResponse(ints, intents, correct_msg)
        else:
            res = random.choice(default_responses)

        print(ints)
        text.clear()
        flag = False
    else:
        if correct_msg == input_msg: # the input is correctly spelled
            ints = predict_class(input_msg, model)
            print(correct_msg, ints)
            res = getResponse(ints, intents, correct_msg)
            text.clear()
            flag = False
        else:
            text.append(correct_msg)
            res = f"Sorry, Did you mean \"{correct_msg}\" instead of \"{input_msg}\"? Please enter yes or no."
            flag = True

    print("Previous Context:", previous_context, " Current Context: ", current_context)

    return res

#### REFERENCE OF HTML, GOOGLE API

from flask import Flask, session, abort, redirect, request, render_template
import logging

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = "secret" #OAuth 2.0 secret key
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
GOOGLE_CLIENT_ID = "327218355433-fi8hnego4ul4jgim97venfov1hpp8cql.apps.googleusercontent.com"  #enter your client id you got from Google console
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")  #set the path to where the .json file you got Google console is

flow = Flow.from_client_secrets_file(  #Flow is OAuth 2.0 a class that stores all the information on how we want to authorize our users
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],  #here we are specifing what do we get after the authorization
    redirect_uri="http://127.0.0.1:5000/callback"  #and the redirect URI is the point where the user will end up after the authorization
)

def login_is_required(function):  #a function to check if the user is authorized or not
    def wrapper(*args, **kwargs):
        if "google_id" not in session:  #authorization required
            return abort(401)
        else:
            return function()

    return wrapper

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/login")  #the page where the user can login
def login():
    authorization_url, state = flow.authorization_url()  #asking the flow class for the authorization (login) url
    session["state"] = state
    return redirect(authorization_url)
@app.route("/callback")  #this is the page that will handle the callback process meaning process after the authorization
def callback():
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)  #state does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID,
        clock_skew_in_seconds=0
    )

    session["google_id"] = id_info.get("sub")  #defing the results to show on the page
    session["name"] = id_info.get("name")
    return redirect("/chat")  #the final page where the authorized users will end up

@app.route("/chat")
def chat():
    return render_template("chatbox.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/logout")  #the logout page and function
def logout():
    session.clear()
    return redirect("/")

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request')
    return 'An internal error occurred', 500

if __name__ == "__main__":
    app.run()