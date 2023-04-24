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
import re
import string
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import re
import openai

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
current_context = []
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
    p = bow(sentence, words,show_details=False)
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
                print(current_context)
                return random.choice(possible_responses)
            else:
                previous_context = current_context
                current_context = i['context']
                return random.choice(possible_responses)    

    define = definition(msg)
    if define != None:
        return define

    return random.choice(default_responses)

def clean_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

# get the definition of words that is not available in the intents.json
def get_definition(input_msg): # using WordNet
    # get the word from the input
    cleaned_tokens = clean_tokens(nltk.word_tokenize(input_msg))
    if cleaned_tokens:
        word = cleaned_tokens[0]
        # get definition
        synsets = wordnet.synsets(word)
        if synsets:
            definition = synsets[0].definition()
            return f"The word {word} means " + definition + "."
    return f"none"

# get definition using openai
def definition(str):
    openai.api_key = "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"
    cleaned_tokens = clean_tokens(nltk.word_tokenize(str))
    if cleaned_tokens:
        word = cleaned_tokens[0]
        prompt = f"Explain {word} to a primary school student."
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
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
    if flag:
        output_word=[correct_msg for correct_msg in text]
        output_txt=" ".join(output_word)
        print("current input: ", output_txt)

        if correct_msg == "yes" or correct_msg in output_txt:
            ints = predict_class(output_txt, model)
            res = getResponse(ints, intents, output_txt)
            print(ints)
            text.clear()
            flag = False
        elif correct_msg == "no":
            res = random.choice(res)
            flag = False
        else:
            res = random.choice(default_responses)
            flag = False
    else:
        if correct_msg == input_msg: # the input is correctly spelled
            if correct_msg == "type" or correct_msg == "types":
                correct_msg = "types " + ' '.join(current_context)
            
            if correct_msg == "example" or correct_msg == "examples":
                correct_msg = "examples " + ' '.join(current_context)
            ints = predict_class(correct_msg, model)
            print(correct_msg, ints)
            res = getResponse(ints, intents, correct_msg)

            text.clear()
            flag = False
        else:
            text.append(correct_msg)
            res = f"Sorry, Did you mean \"{correct_msg}\" instead of \"{input_msg}\"? Please enter yes or no."
            flag = True

        if "pos" in input_msg:
            pos = check_POS(input_msg) # check if the user wants to check the POS of a word
            if pos != False:
                print("pos: ", pos)
                split_msg = input_msg.split()
                split_msg.remove(pos) # remove the pos to not confuse data cleaning
                new_msg = " ".join(split_msg)
                return get_POS(new_msg, pos)


        

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