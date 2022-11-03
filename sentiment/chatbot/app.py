from flask import Flask
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, url_for, redirect
from pymongo import MongoClient
from .response_generator import generate_response
from transformers import AutoTokenizer, AutoModel, pipeline,AutoModelForQuestionAnswering


app = Flask(__name__)



@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    response = ""
    if message:     
        response = generate_response(message)
        response= str(response)
        if response==".":
            response="Sorry! I didn't get it, please try to be more precise."
        
        return response
    else:
        return "Missing Data!"



if __name__ == "__main__":
    
    app.run(host='localhost', debug=True,port=5000)
