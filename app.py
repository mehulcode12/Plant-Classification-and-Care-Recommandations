from flask import Flask, render_template, request
from keras.models import load_model
# from keras.preprocessing.image import 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from os import environ 

OPENAI_API_KEY = environ.get('OPENAI_API_KEY')


load_dotenv()  



# app = Flask(__name__)
# dic = {0: 'aloevera', 1: 'hibiscous', 2: 'rose', 3: 'sugarcane'}
# model = tf.keras.models.load_model("mehul-pbl-plant1.h5")

app = Flask(__name__)
dic = {0: 'Aloevera', 1: 'Hibiscous', 2: 'Rose', 3: 'Sugarcane'}
model = tf.keras.models.load_model("mehul-pbl-plant1.h5")


# def import_n_pred(image_path, model):
#     size = (64, 64)
#     image = Image.open(image_path)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)
#     img = np.asarray(image)
#     reshape = img[np.newaxis,...]
#     pred = model.predict(reshape)
#     max_index = np.argmax(pred)
#     return dic[max_index]




def import_n_pred(image_path, model):
    size = (64, 64)
    image = Image.open(image_path)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image data
    reshape = img[np.newaxis, ...]
    pred = model.predict(reshape)
    max_index = np.argmax(pred)
    return dic[max_index]


import openai

# Get the API key from the environment variable
api_key = os.environ["OPENAI_API_KEY"]

# Set the API key in the OpenAI client
openai.api_key = api_key


def get_care_recommendations(plant_name):
    prompt = f"What are the care recommendations in 200 words for {plant_name} plant? consider the points like watering, sunlight, season, and this plant is from india and give care recommendations using modern techniques. and scientific language and more attractive text "
    model = "text-davinci-002"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    care_recommendations = response.choices[0].text.strip()
    return care_recommendations

@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = 'static/' + img.filename
        img.save(img_path)

        predicted_plant = import_n_pred(img_path, model)
        care_recommendations = get_care_recommendations(predicted_plant)
        print(f"Predicted plant: {predicted_plant}")
        print(f"Care recommendations: {care_recommendations}")

    return render_template('model.html', prediction=predicted_plant, care_recommendations=care_recommendations, img_path=img_path)

# def import_n_pred(img_path, model):
#     img = Image.open(img_path)
#     img = img.resize((128, 128))
#     img = np.asarray(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     print(pred)
#     return dic[np.argmax(pred)]


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def page_model():
    return render_template('model.html')

# @app.route('/submit', methods=['GET', 'POST'])
# def get_output():
#     if request.method == 'POST':
#         img = request.files['my_image']

#         img_path = 'static/' + img.filename
#         img.save(img_path)

#         p = import_n_pred(img_path, model)
#         print(p)

#     return render_template('model.html', prediction=p, img_path=img_path)



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)