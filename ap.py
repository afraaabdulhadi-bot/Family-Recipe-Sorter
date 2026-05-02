from flask import Flask, render_template, request, jsonify
import base64
import os
from inference_sdk import InferenceHTTPClient
from openai import OpenAI

app = Flask(__name__)

# 🔐 Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# 🤖 Clients
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

ai_client = OpenAI(api_key=OPENAI_API_KEY)

# 📖 Recipes
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
info_cache = {}

@app.route("/")
def home():
    return render_template("index.html")

# 📷 Detect Image
@app.route("/detect", methods=["POST"])
def detect():
    global current_label, current_index

    try:
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])

        with open("frame.jpg", "wb") as f:
            f.write(image_data)

        if ROBOFLOW_API_KEY:
            result = CLIENT.infer("frame.jpg", model_id="smart-family-recipe-sorter/1")

            if result.get("predictions"):
                label = result["predictions"][0]["class"]

                if label in ["loomi", "dried_lime", "black_lime"]:
                    label = "lime"

                if label in recipes:
                    current_label = label
                    current_index = 0

    except Exception as e:
        print("Detect error:", e)

    return jsonify(recipes[current_label][current_index])

# ➡️ Next Recipe
@app.route("/next")
def next_recipe():
    global current_index
    current_index = (current_index + 1) % len(recipes[current_label])
    return jsonify(recipes[current_label][current_index])

# 🤖 AI Info
@app.route("/info")
def info():
    dish = recipes[current_label][current_index]["en"]

    if dish in info_cache:
        return jsonify({"text": info_cache[dish]})

    try:
        response = ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""
Explain {dish}:
1- What is it
2- Relation to UAE culture
3- Main ingredients

Arabic + English simple.
"""
            }]
        )

        text = response.choices[0].message.content
        info_cache[dish] = text

    except Exception as e:
        print("AI error:", e)
        text = "❌ AI not working"

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run()