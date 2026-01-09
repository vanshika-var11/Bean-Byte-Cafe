from time import time
import streamlit as st
import pandas as pd
import time
import random
from recommender import get_recommendations, evaluate_system
import plotly.express as px

MENU_KB = {
    "Veg Burger": {"price": 120, "ingredients": "Bun, Veg Patty, Lettuce, Tomato, Cheese, Mayo"},
    "Chicken Burger": {"price": 150, "ingredients": "Bun, Chicken Patty, Lettuce, Cheese, Sauce"},
    "Paneer Roll": {"price": 130, "ingredients": "Paneer, Roti, Onion, Capsicum, Spices"},
    "Chicken Roll": {"price": 140, "ingredients": "Chicken, Roti, Onion, Capsicum, Spices"},
    "Margherita Pizza": {"price": 220, "ingredients": "Pizza Base, Tomato Sauce, Mozzarella"},
    "Pepperoni Pizza": {"price": 260, "ingredients": "Pizza Base, Pepperoni, Cheese"},
    "Pasta Alfredo": {"price": 210, "ingredients": "Pasta, Cream, Garlic, Cheese"},
    "French Fries": {"price": 90, "ingredients": "Potato, Salt, Oil"},
    "Chocolate Shake": {"price": 110, "ingredients": "Milk, Chocolate Syrup, Ice Cream"},
    "Cold Coffee": {"price": 100, "ingredients": "Coffee, Milk, Sugar, Ice"},
    "Caesar Salad": {"price": 140, "ingredients": "Lettuce, Croutons, Parmesan, Caesar Dressing"},
    "Momos": {"price": 120, "ingredients": "Flour, Veg/Chicken Filling, Spices"},
    "Veg Biryani": {"price": 180, "ingredients": "Rice, Vegetables, Spices"},
    "Chicken Biryani": {"price": 220, "ingredients": "Rice, Chicken, Spices"},
}

CAFE_INFO = {
    "opening_time": "10:00 AM",
    "closing_time": "11:00 PM",
    "location": "Bean & Byte Café, Main Street23, Cityville,Mumbai",
}

MENU_IMAGES = {
    "Veg Burger": "https://media.istockphoto.com/id/905921648/photo/healthy-baked-sweet-potato-burger-with-whole-grain-bun-guacamole-vegan-mayonnaise-and.jpg?s=612x612&w=0&k=20&c=7mrEKIn1cQbN0Q6k3ONHPsaN3jmsYz_X7YExQIOqG9A=",
    "Chicken Burger": "https://www.shutterstock.com/image-photo/crispy-chicken-burger-isolation-on-600nw-2626642199.jpg",
    "Paneer Roll": "https://media.istockphoto.com/id/1209898811/photo/chapati-wrap-with-cheese-vegetarian-food.jpg?s=612x612&w=0&k=20&c=nSbQOQrOq4W2VDYlazvsX9BERrqPekAM5KIEMf0kwCU=",
    "Chicken Roll": "https://i.ytimg.com/vi/NfQ7p_LzpUA/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDenQlVPqAu4VbSianFVxHPptlW8g",
    "Margherita Pizza": "https://thumbs.dreamstime.com/b/pizza-margherita-27409337.jpg",
    "Pepperoni Pizza": "https://media.istockphoto.com/id/521403691/photo/hot-homemade-pepperoni-pizza.jpg?s=612x612&w=0&k=20&c=PaISuuHcJWTEVoDKNnxaHy7L2BTUkyYZ06hYgzXmTbo=",
    "Pasta Alfredo": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq-QD4Yx_TZGSrdRFFCH0_PzoGcfHp5g1sNQ&s",
    "French Fries": "https://media.istockphoto.com/id/1443993866/photo/french-fries-with-ketchup-and-cocktail-sauce.jpg?s=612x612&w=0&k=20&c=URpOsc5tds8tOfxK4ZO3Tkx6mwLho7fL_pTBSNdziBU=",
    "Chocolate Shake": "https://media.istockphoto.com/id/1291175021/photo/chocolate-milkshake.jpg?s=612x612&w=0&k=20&c=LkPVfYUcIdM85m9FI6QrAjXyzvOxrdhBEkmcPmlbTYw=",
    "Cold Coffee": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKIEXCoefXcGXZONVk48FxHTD2qjQqYbfwWA&s",
    "Caesar Salad": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHX1wizyBixkrUkyNdAw-dYjLaA-f2vnHGSg&s",
    "Momos": "https://img.freepik.com/premium-photo/tibetian-dumplings-momo-with-chicken-meat-vegetables_1472-119353.jpg?semt=ais_hybrid&w=740&q=80",
    "Veg Biryani": "https://www.shutterstock.com/image-photo/veg-biryani-tempting-600nw-2583130989.jpg",
    "Chicken Biryani": "https://thumbs.dreamstime.com/b/vibrant-appetizing-diagonal-shot-traditional-indian-chicken-biryani-served-alongside-rich-flavorful-curry-fresh-mixed-422599560.jpg"
}


# Page Config
st.set_page_config("Bean & Byte Café", page_icon="☕", layout="wide")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins&display=swap');

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #f5e6d3, #e8c39e, #b08968);
    animation: cafeGlow 12s ease infinite;
}

@keyframes cafeGlow {
    0% { filter: brightness(1); }
    50% { filter: brightness(1.07); }
    100% { filter: brightness(1); }
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3e2723, #4e342e);
    color: white;
}

.big-title {
    font-family: 'Playfair Display', serif;
    font-size: 60px;
    color: #5a3825;
}

.sub-title {
    font-family: 'Poppins', sans-serif;
    font-size: 22px;
    color: #8b5e3c;
}

.card {
    background: rgba(255, 250, 240, 0.95);
    backdrop-filter: blur(6px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(90, 56, 37, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* -------- FIX INPUTS & DROPDOWNS -------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #3e2723 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #3e2723 !important;
}

/* Slider labels */
div[data-testid="stSlider"] span {
    color: #3e2723 !important;
}

/* Theme dropdown in sidebar */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #3e2723 !important;
}

/* Remove ghost dark overlay */
.css-1d391kg, .css-1v0mbdj {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Global text color fix */
html, body, [class*="st-"], p, span, label, div {
    color: #3e2723 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #f5e6d3 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"], 
[data-testid="stMetricValue"] {
    color: #3e2723 !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #4e342e !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.loader {
  border: 6px solid #f3f3f3;
  border-top: 6px solid #6f4e37;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: auto;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ---------- THEME DROPDOWN FIX ---------- */
div[data-baseweb="select"] {
    background: white !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #3e2723 !important;
}

/* ---------- BUTTON VISIBILITY ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6f4e37, #b08968) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 17px !important;
    box-shadow: 0px 8px 18px rgba(111, 78, 55, 0.5);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 12px 24px rgba(111, 78, 55, 0.8);
}

/* ---------- RECOMMENDATION CARDS ---------- */
.stAlert {
    background: rgba(255,255,255,0.85) !important;
    border-radius: 14px !important;
    font-size: 18px !important;
    box-shadow: 0px 8px 20px rgba(90,56,37,0.3);
    animation: pop 0.4s ease;
}

@keyframes pop {
  from { transform: scale(0.9); opacity:0 }
  to { transform: scale(1); opacity:1 }
}

/* ---------- DASHBOARD METRIC SIZE ---------- */
[data-testid="stMetricValue"] {
    font-size: 42px !important;
}

[data-testid="stMetricLabel"] {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* ---------- CHART LABEL SIZE ---------- */
.plotly-graph-div text {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class='card'>
    <div class='big-title'>☕ Bean & Byte Café</div>
    <div class='sub-title'>AI Powered Food Recommendation System</div>
</div>
""", unsafe_allow_html=True)

# Load Data
df = pd.read_csv("customers.csv")

# Sidebar Navigation
theme = st.sidebar.selectbox("🎨 Theme", ["Café",  "Dark"])
menu = st.sidebar.radio("Navigation", ["Customer Portal", "Dashboard", "Menu", "Chef AI"])
# ---------------- THEME SWITCHER ----------------
theme_map = {
    "Café": ("#b8611b", "#5a3825", "#fffaf3"),
    "Dark": ("#0e1117", "#f5f5f5", "#1f2933")
}

bg, text, card = theme_map[theme]

st.markdown(f"""
<style>
body {{
    background: {bg} !important;
}}
.big-title {{
    color: {text} !important;
}}
.sub-title {{
    color: {text} !important;
}}
.card {{
    background: {card} !important;
}}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------



# ---------- CUSTOMER PORTAL ----------
if menu == "Customer Portal":
    st.header("Customer Experience")

    name = st.text_input("Enter Customer Name")

    taste = st.slider("Taste Rating", 1, 5, 3)
    price = st.slider("Price Rating", 1, 5, 3)
    category = st.selectbox("Taste Preference", ["spicy", "sweet", "cheesy", "healthy", "neutral"])
    
    st.markdown("### 🔴 Live AI Recommendation Feed")

    live_box = st.empty()
    import random, time

    live_samples = [
       "Customer 0042 loves Cold Coffee ☕",
       "Recommending Veg Burger 🍔",
       "Hot seller today: Margherita Pizza 🍕",
       "New customer detected 👤",
       "Combo applied: Burger + Fries 🍟",
   ]

    if "live_running" not in st.session_state:
        st.session_state.live_running = True

    def run_live_feed():
        while st.session_state.live_running:
           live_box.info(random.choice(live_samples))
           time.sleep(2)

    if st.button("Get Recommendations"):
        placeholder = st.empty()

        with placeholder.container():
            with st.spinner("Brewing your perfect cup... ☕"):
               time.sleep(1.2)
               recs = get_recommendations(name, taste, price, category)

        placeholder.empty()

        st.subheader("🎯 Recommended For You")

        for item in recs["recommendations"]:
            price = MENU_KB[item]["price"]
            st.markdown(f"""
            <div class="card">
                <h3>🍽 {item}</h3>
                <p>💰 ₹{price}</p>
                <p>🥤 Combo Suggestion: {item} + Cold Coffee @ ₹{price+40}</p>
            </div>
            """, unsafe_allow_html=True)



# ---------- DASHBOARD ----------
elif menu == "Dashboard":
    st.header("Café Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Customers", df["customer_id"].nunique())
        st.metric("Total Visits", len(df))


    with col2:
        st.metric("Top Item", df["food_ordered"].value_counts().idxmax())
        st.metric("Avg Rating", round(df["avg_rating"].mean(), 2))
    
    top_items = df["food_ordered"].value_counts().head(10).reset_index()
    top_items.columns = ["Item", "Orders"]

    fig = px.bar(
        top_items,
        x="Item",
        y="Orders",
        text="Orders",
        animation_frame=None,
        title="Top 10 Selling Items",
    )

    fig.update_traces(marker_color="#6f4e37")
    fig.update_layout(
       title_font_color="#4e342e",
       xaxis_title_font_color="#4e342e",
       yaxis_title_font_color="#4e342e",
       xaxis=dict(
        tickfont=dict(color="#3e2723"),
        title_font=dict(color="#4e342e")
    ),
       yaxis=dict(
        tickfont=dict(color="#3e2723"),
        title_font=dict(color="#4e342e")
    ),
    legend=dict(font=dict(color="#3e2723")),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    transition_duration=1000
)

#     fig.update_layout(
#        transition_duration=1000,
#        plot_bgcolor="rgba(0,0,0,0)",
#        paper_bgcolor="rgba(0,0,0,0)"
#    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📈 Explore Trends")
    choice = st.selectbox("Choose view", ["Top Items", "Low Rated Items"])

    if choice == "Low Rated Items":
       low = df.sort_values("avg_rating").head(5)
       st.dataframe(low)

#-----------MENU-------------
elif menu == "Menu":
    st.header("🍽 Today's Menu")

    cols = st.columns(3)
    i = 0

    for item, info in MENU_KB.items():
        with cols[i % 3]:
            with st.container():
                st.markdown(f"### {item}")
                st.image(MENU_IMAGES.get(item, ""), use_container_width=True)

                st.markdown(f"**₹ {info['price']}**")
                
        i += 1

    if "selected_item" in st.session_state:
        item = st.session_state.selected_item
        info = MENU_KB[item]

        st.markdown("---")
        st.subheader(f"🍽 {item}")
        st.image(MENU_IMAGES.get(item, ""), use_container_width=True)

        st.markdown(f"### 💰 Price: ₹ {info['price']}")
        





elif menu == "Chef AI":
    st.header("🧑‍🍳 Chef AI Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("Ask about menu, ingredients, price, opening time...")

    if user_input:
        question = user_input.lower()
        st.session_state.chat.append(("user", user_input))

        reply = "I'm here to help! 😊"

        # ⏰ Café timings
        if "time" in question or "open" in question:
            reply = f"We are open from {CAFE_INFO['opening_time']} to {CAFE_INFO['closing_time']}."

        # 📍 Location
        elif "where" in question or "location" in question:
            reply = f"Our café is located at {CAFE_INFO['location']}."

        # 🍽 Menu item queries
        else:
            for item, info in MENU_KB.items():
                if item.lower() in question:
                    if "price" in question:
                        reply = f"The price of {item} is ₹{info['price']}."
                    elif "ingredient" in question or "made" in question:
                        reply = f"{item} is made with: {info['ingredients']}."
                    else:
                        reply = f"{item} costs ₹{info['price']} and contains {info['ingredients']}."
                    break

        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").write(reply)


