import streamlit as st
from inference_sdk import InferenceHTTPClient
from openai import OpenAI
import os

# 🔑 Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ROBOFLOW_KEY = os.getenv("ROBOFLOW_API_KEY")

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_KEY
)

ai_client = OpenAI(api_key=OPENAI_KEY)

# 🍽️ Recipes (نفس الكود تبعك)
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

# 🧠 Session state (بدل global)
if "current_label" not in st.session_state:
    st.session_state.current_label = "dates"

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.title("🍽️ UAE Recipe AI")

# 📷 Camera
img = st.camera_input("Take a picture")

# 📷 Detect
if img is not None:
    with open("frame.jpg", "wb") as f:
        f.write(img.getbuffer())

    try:
        result = CLIENT.infer("frame.jpg", model_id="smart-family-recipe-sorter/1")

        if result["predictions"]:
            label = result["predictions"][0]["class"]

            # 🔥 حل اللومي
            if label in ["loomi", "dried_lime", "black_lime"]:
                label = "lime"

            if label in recipes:
                st.session_state.current_label = label
                st.session_state.current_index = 0

    except:
        st.error("Detection error")

# 📌 عرض الوصفة الحالية
current = recipes[st.session_state.current_label][st.session_state.current_index]

st.subheader(current["en"])
st.write(current["ar"])

# 🖼️ صورة (إذا موجودة)
if os.path.exists(current["img"]):
    st.image(current["img"])

# ➡️ Next button
if st.button("Next"):
    st.session_state.current_index = (
        st.session_state.current_index + 1
    ) % len(recipes[st.session_state.current_label])

# 🧠 AI Info
if st.button("Explain Dish"):
    try:
        prompt = f"""
        Explain {current['en']}:
        1- What is it
        2- Relation to UAE culture
        3- Main ingredients
        Arabic + English simple.
        """

        response = ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)

    except:
        st.error("AI error")