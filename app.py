from flask import Flask, render_template, request, jsonify
import base64
import os 
from inference_sdk import InferenceHTTPClient
from openai import OpenAI

app = Flask(__name__)

# 🔴 حطي مفاتيحك
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ROBOFLOW_KEY = os.getenv("ROBOFLOW_API_KEY")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_KEY
)

ai_client = OpenAI(api_key=OPENAI_KEY)

recipes = {
    "dates": [
        {"ar": "تمر محشي", "en": "Stuffed dates", "img": "stuffed_dates.jpg"},
        {"ar": "كيك التمر", "en": "Date cake", "img": "date_cake.jpg"},
        {"ar": "دبس التمر", "en": "Date syrup", "img": "date_syrup.jpg"},
        {"ar": "لقيمات", "en": "Luqaimat", "img": "luqaimat.jpg"}
    ],
    "saffron": [
        {"ar": "رز بالزعفران", "en": "Saffron rice", "img": "saffron_rice.jpg"},
        {"ar": "شاي الزعفران", "en": "Saffron tea", "img": "saffron_tea.jpg"},
        {"ar": "حليب بالزعفران", "en": "Saffron milk", "img": "saffron_milk.jpg"},
        {"ar": "كرك", "en": "Karak", "img": "karak.jpg"}
    ],
    "cardamom": [
        {"ar": "قهوة عربية", "en": "Arabic coffee", "img": "arabic_coffee.jpg"},
        {"ar": "كيك الهيل", "en": "Cardamom cake", "img": "cardamom_cake.jpg"},
        {"ar": "رز بالهيل", "en": "Cardamom rice", "img": "cardamom_rice.jpg"},
        {"ar": "بلا ليط", "en": "Balaleet", "img": "balaleet.jpg"}
    ],
    "lime": [
        {"ar": "شوربة لومي", "en": "Lime soup", "img": "lime_soup.jpg"},
        {"ar": "أرز باللومي", "en": "Lime rice", "img": "lime_rice.jpg"},
        {"ar": "مجبوس", "en": "Machboos", "img": "machboos.jpg"},
        {"ar": "سمك باللومي", "en": "Fish lime", "img": "fish_lime.jpg"}
    ]
}

current_label = "dates"
current_index = 0

@app.route("/")
def home():
    return render_template("index.html")

# 📷 Detect
@app.route("/detect", methods=["POST"])
def detect():
    global current_label, current_index

    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])

    with open("frame.jpg", "wb") as f:
        f.write(image_data)

    try:
        result = CLIENT.infer("frame.jpg", model_id="smart-family-recipe-sorter/1")

        if result["predictions"]:
            label = result["predictions"][0]["class"]

            # 🔥 حل اللومي
            if label in ["loomi", "dried_lime", "black_lime"]:
                label = "lime"

            if label in recipes:
                current_label = label
                current_index = 0

    except:
        pass

    return jsonify(recipes[current_label][current_index])

# ➡️ Next
@app.route("/next")
def next_recipe():
    global current_index, current_label
    current_index = (current_index + 1) % len(recipes[current_label])
    return jsonify(recipes[current_label][current_index])

# 🧠 ChatGPT Info
@app.route("/info/<dish>")
def info(dish):
    try:
        prompt = f"""
        Explain {dish}:

        1- What is it
        2- Relation to UAE culture
        3- Main ingredients

        Arabic + English simple.
        """

        response = ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return jsonify({"text": response.choices[0].message.content})

    except:
        return jsonify({"text": "Error"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)