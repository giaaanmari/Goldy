#### IMPORT LIBRARIES

import os
import requests
import pathlib
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
import pickle
import enchant
import string
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import openai
import re

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
similarity_threshold = 0.5

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
    sentence_words = [w for w in (nltk.word_tokenize(sentence))]
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in set(stopwords.words('english'))]
    print("(clean_up_sentence) sentence words: ", sentence_words)
    return sentence_words

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
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
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
    
    return random.choice(default_responses)

def clean_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

def definition(str):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Reply to this input \"{str}\" as Goldy, a child friendly assistant that helps with English subject",
        max_tokens=100
    )
    if response.choices[0].text:
        return response.choices[0].text.strip()        
    return f"none"

# check if the input contains pos 
def check_POS(str):
    keywords = words
    for word in str.split():
        if word in keywords:
            # check if there is another word other than the POS
            split_str = str.split()
            split_str.remove(word)
            new_str = " ".join(split_str)
            cleaned_tokens = clean_tokens(nltk.word_tokenize(new_str))
            if cleaned_tokens:
                return word
    return False

# code if the user wants to know the POS of a word (not yet done here) 
def get_POS(str, pos):    
    cleaned_tokens = clean_tokens(nltk.word_tokenize(str))
    if cleaned_tokens:
        word = cleaned_tokens[0]
    
        synsets = wordnet.synsets(word)
        pos_list = [synset.pos() for synset in synsets]
        print(pos_list)
        if pos[0] in pos_list:
            return f"Yes. The word {word} is a/an {pos}."
        else: 
            return f"No. The word {word} is not a/an {pos}."

def spell(match):
    word = match.group(0)
    return dictionary.suggest(word)[0] if not dictionary.check(word) and dictionary.suggest(word) else word

def checkKey(str):
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
    if isExample:
        key = "example"
        tempKey = "".join(isExample)
        str = [element for element in user_input if element != isExample]
    if isType:
        key = "type"
        tempKey = "".join(isType)
    str = [element for element in user_input if element != tempKey]

    if len(str) == 0:
        return key
    else:
        return None

flag = False
text = []
def chatbot_response(input_msg):
    global text
    global flag
    global context
    global default_responses

    res = ["Okay. May you please clarify what you're asking or provide more information so I can better assist you?",
           "Okay. Could you please clarify your question so I can assist you better?",
           "Okay. Could you please rephrase your question or provide more details so I can better understand how to help you?"]

    input_msg = input_msg.lower()
    correct_msg = re.sub(r"\w+", spell, input_msg) # correct the spelling of the input_msg
    
    if current_context != None:
        key = checkKey(correct_msg) # check if the user_input contains keywords that asks for example or type without typing the main context
        # verify is there is a key and the current context must have a value (means this is not the first input)
        if key != None:
            correct_msg = key + ' of ' + current_context[0]
            input_msg = correct_msg # change the value of input_msg to proceed to flag = False and get response from `if correct_msg == input_msg`
            print("new message: ", correct_msg)

    if flag:
        output_word=[correct_msg for correct_msg in text] # create a list of correct_msg everytime the user input a misspelled words
        output_txt=" ".join(output_word) 
        if correct_msg == "yes" or correct_msg in output_txt:
            ints = predict_class(output_txt, model) # output_txt is the last correct spelling. we are sending it to find the intent because correct_msg is "yes" and there's no tag for yes
            res = getResponse(ints, intents, output_txt)
            print(ints)
            text.clear()
            flag = False
        elif correct_msg == "no":
            res = random.choice(res)
            flag = False
        elif correct_msg == input_msg: # this is when the user decide to input another message with correct spelling while the flag is True
            ints = predict_class(correct_msg, model)
            res = getResponse(ints, intents, correct_msg)
            print(ints)
            text.clear()
        else:
            res = random.choice(default_responses)
            flag = False
    else:
        if correct_msg == input_msg: # the input is correctly spelled
            ints = predict_class(correct_msg, model)
            print(correct_msg, ints)
            res = getResponse(ints, intents, correct_msg)
            print("Previous Context:", previous_context, " Current Context: ", current_context)
            text.clear()
            flag = False
        else:
            text.append(correct_msg)
            res = f"Sorry, Did you mean \"{correct_msg}\" instead of \"{input_msg}\"? Please enter yes or no."
            flag = True
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
    app.run(host='0.0.0.0')