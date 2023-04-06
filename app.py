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
nltk.download('popular')
nltk.download('words')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
import pickle
from autocorrect import Speller
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
context = None
similarity_threshold = 0.6
default_responses = ["I'm sorry, I didn't quite understand what you said. Can you please try asking me again in a different way?",
                     "I'm sorry, I don't have the answer to that question right now. But don't worry, I'll keep learning and hopefully, I'll be able to help you with your question soon.",
                     "Hmm, I'm not quite sure what you're asking. Can you please give me more information or context about your question?"]
spell = Speller(lang='en')
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
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    global default_responses
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            if 'context' not in intents or 'context' in intents and intents['context'] == context:
                possible_responses = i['responses']
                if 'context' in intents:
                    context = intents['context']
                else:
                    context = None                        
                result = random.choice(possible_responses)
            else:
                result = random.choice(default_responses)
            break
        else:
            result = random.choice(default_responses)
    return result

flag = False
text = []

def chatbot_response(input_msg):
    input_msg = input_msg.lower()
    global text
    global flag
    
    correct_msg = spell(input_msg)

    if flag:
        output_word=[correct_msg for correct_msg in text]
        output_txt=" ".join(output_word)

        if correct_msg == "yes" or correct_msg in output_txt:
            ints = predict_class(output_txt, model)
            res = getResponse(ints, intents)
            flag = False
        elif correct_msg == "no":
            res = f"Okay. Could you please clarify your question so I can assist you better?"
            flag = False
        else:
            res = f"Sorry, I am still learning. Please enter your message again."
            flag = False
        
    else:
        text.append(correct_msg)
        if correct_msg == input_msg:
            ints = predict_class(input_msg, model)
            res = getResponse(ints, intents)
            text.clear()
            flag = False
        else:
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
    app.run()