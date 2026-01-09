import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import random

# ---------- CONFIG ----------
DAYS = 16
UNIQUE_CUSTOMERS = 500 # Define a fixed number of unique customers
VISITS_PER_DAY = 500 # Total visits per day
TOTAL_ROWS = DAYS * VISITS_PER_DAY

# Indian names dataset
MALE_NAMES = ["Raj","Amit","Vikram","Rohan","Anil","Suresh","Manish","Deepak","Arjun","Karan",
              "Harsh","Mohit","Nitin","Rajat","Sunny","Tarun","Yogesh","Hemant","Jatin","Nikhil"]

FEMALE_NAMES = ["Priya","Neha","Anjali","Pooja","Ritika","Sakshi","Divya","Anu","Meera","Kavita",
                "Tanya","Shruti","Komal","Simran","Aisha","Sonam","Pallavi","Kritika","Alisha","Rekha"]

SURNAMES = ["Sharma","Varshney","Saini","Mittal","Bansal","Tyagi", "Verma", "Gupta", "Yadav", "Singh", "Agarwal", "Chauhan", "Khan", "Patel", "Mehta"]

MENU_ITEMS = [
    "Veg Burger", "Chicken Burger", "Paneer Roll", "Chicken Roll",
    "Margherita Pizza", "Pepperoni Pizza", "Pasta Alfredo",
    "French Fries", "Spring Rolls", "Chocolate Shake",
    "Cold Coffee", "Caesar Salad","Momos","Veg Kabab","Vegetable Chowmein",
    "Veg Biryani","Chicken Biryani","Soft Drinks","Mixed Fruit Juice","Mocktails"
]

def random_phone():
    return "9" + "".join(str(random.randint(0,9)) for _ in range(9))

def random_time():
    return time(
        hour=np.random.choice([11,12,13,14,17,18,19,20,21]),
        minute=random.randint(0,59)
    )

first_name_count = {}

def random_indian_name():
    global first_name_count

    all_first_names = MALE_NAMES + FEMALE_NAMES

    # If dictionary is empty → initialize all names with 0
    if len(first_name_count) == 0:
        first_name_count = {name: 0 for name in all_first_names}

    # If ALL names reached limit → reset counters
    if all(first_name_count[name] >= 3 for name in all_first_names):
        first_name_count = {name: 0 for name in all_first_names}

    # Pick only names that still have <3 count
    available = [name for name in all_first_names if first_name_count[name] < 3]

    # Randomly select from available names
    first = random.choice(available)

    first_name_count[first] += 1
    last = random.choice(SURNAMES)

    return f"{first} {last}"

rows = []
start_date = datetime.today().date() - timedelta(days=DAYS)

visit_id = 1

# Generate a pool of unique customers first
CUSTOMER_POOL = []
for i in range(UNIQUE_CUSTOMERS):
    CUSTOMER_POOL.append({
        "customer_id": f"C{i+1:04d}",
        "customer_name": random_indian_name(),
        "age": random.randint(16, 65),
        "phone": random_phone()
    })

for d in range(DAYS):
    current_date = start_date + timedelta(days=d+1)

    for _ in range(VISITS_PER_DAY):
        # Randomly pick a customer from the pool for each visit
        if random.random() < 0.75:
          customer_info = random.choice(CUSTOMER_POOL[:250])
        else:
          customer_info = random.choice(CUSTOMER_POOL)
        cust_id = customer_info["customer_id"]
        customer_name = customer_info["customer_name"]
        age = customer_info["age"]
        phone = customer_info["phone"]

        arrival = datetime.combine(current_date, random_time())
        # -------------------------------
        # CUSTOMER FOOD PREFERENCE MEMORY
        # -------------------------------
        if "favorite_food" not in customer_info:
            customer_info["favorite_food"] = random.choice(MENU_ITEMS)

        # 70% chance customer repeats preference
        if random.random() < 0.94:
            food = customer_info["favorite_food"]
      
        else:
            food = random.choice(MENU_ITEMS)

        # Slowly evolve taste over time
        if random.random() < 0.02:
            customer_info["favorite_food"] = random.choice(MENU_ITEMS)
        taste = random.randint(1,5)
        price = random.randint(1,5)
        avg_rating = round((taste + price) / 2, 2)

        party_size = np.random.choice([1,2,3,4,5], p=[0.35,0.4,0.15,0.07,0.03])

        rows.append({
            "visit_id": f"V{visit_id:05d}",
            "customer_id": cust_id,
            "customer_name": customer_name,
            "age": age,
            "phone": phone,
            "date": current_date,
            "arrival_time": arrival,
            "food_ordered": food,
            "taste_rating": taste,
            "price_rating": price,
            "avg_rating": avg_rating,
            "how_many_with_him": party_size
        })
        visit_id += 1

# Convert to CSV
df = pd.DataFrame(rows)
df.to_csv("customers.csv", index=False)

print("✅ CSV generated successfully: customers.csv")
print(df.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------------
# 1. Load STEP-1 data
# -----------------------------------
df = pd.read_csv("customers.csv")

# -----------------------------------
# 2. Compute per-visit averages
# -----------------------------------
df["Avg_Taste"] = df.groupby(["customer_id", "date"])["taste_rating"].transform("mean")
df["Avg_Price"] = df.groupby(["customer_id", "date"])["price_rating"].transform("mean")

# -----------------------------------
# 3. Features for ML Clustering
# -----------------------------------
features = df[["Avg_Taste", "Avg_Price"]].copy()

# Scale features (IMPORTANT)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# -----------------------------------
# VALIDATION: Check if 7 clusters is optimal
# -----------------------------------
inertia = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, labels))

# Plot validation
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("K")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# Show result clearly
best_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"🔍 Best K according to Silhouette Score: {best_k}")
print("📌 Production system still uses K = 7 (business requirement)")

# -----------------------------------
# 4. K-Means clustering (7 clusters)
# -----------------------------------
kmeans = KMeans(n_clusters=7, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# -----------------------------------
# 5. Track previous visit cluster
# -----------------------------------
df = df.sort_values(["customer_id", "date"])
df["Prev_Cluster"] = df.groupby("customer_id")["Cluster"].shift(1)
df["Cluster_Changed"] = df["Prev_Cluster"] != df["Cluster"]
df["Cluster_Changed"] = df["Cluster_Changed"].fillna(False)

# -----------------------------------
# 6. Save output
# -----------------------------------
df.to_csv("step2_clustered_visits.csv", index=False)

df.head(20)

import pandas as pd

# -----------------------------
# 1. Load STEP-2 clustered file
# -----------------------------
df = pd.read_csv("step2_clustered_visits.csv")

# -----------------------------
# 2. Count item orders per day
# -----------------------------
daily_sales = (
    df.groupby(["date", "food_ordered"])
      .size()
      .reset_index(name="Order_Count")
      .sort_values(["date", "Order_Count"], ascending=[True, False])
)

# Save daily item performance
daily_sales.to_csv("step3_daily_item_sales.csv", index=False)



# -----------------------------
# 3. Compute overall top & least sellers
# -----------------------------
overall_sales = (
    df.groupby("food_ordered")
      .size()
      .reset_index(name="Total_Orders")
      .sort_values("Total_Orders", ascending=False)
)

# Save overall
overall_sales.to_csv("step3_overall_item_sales.csv", index=False)



# -----------------------------
# 4. Identify top 3 and bottom 3
# -----------------------------
top_3 = overall_sales.head(3)
bottom_3 = overall_sales.tail(3)

print("\n🔥 TOP 3 BEST-SELLING ITEMS:")
print(top_3)

print("\n⚠️ BOTTOM 3 LEAST-SELLING ITEMS:")
print(bottom_3)


import pandas as pd
from difflib import get_close_matches
import random

cluster_menu = {
        1: ["French Fries", "Veg Burger", "Cold Coffee"],
        2: ["Pepperoni Pizza", "Chicken Burger", "Pasta Alfredo"],
        3: ["Veg Burger", "Caesar Salad", "Cold Coffee"],
        4: ["Chicken Roll", "Pasta Alfredo", "Pepperoni Pizza"],
        5: ["Paneer Roll", "Chocolate Shake"],
        6: ["Pasta Alfredo", "Chicken Burger"],
        7: ["Margherita Pizza", "Pepperoni Pizza", "Pasta Alfredo", "Chocolate Shake"]
    }

category_items = {
        "spicy": ["Chicken Roll", "Pepperoni Pizza"],
        "sweet": ["Chocolate Shake", "Brownie"],
        "cheesy": ["Cheese Pizza", "Cheese Garlic Bread"],
        "healthy": ["Green Salad", "Caesar Salad"],
        "neutral": ["French Fries", "Cold Coffee"]
    }

combo_map = {
        "Pizza": "Garlic Bread + Cold Drink",
        "Burger": "French Fries + Coke",
        "Pasta": "Garlic Bread",
        "Roll": "Cold Drink",
        "Salad": "Fresh Lime Water"
    }

# -----------------------------
# 1. Load STEP-2 and STEP-3 outputs
# -----------------------------
df = pd.read_csv("step2_clustered_visits.csv")
overall_sales = pd.read_csv("step3_overall_item_sales.csv")

# -----------------------------
# 2. Identify global top & problem items
# -----------------------------
top_items = overall_sales.head(3)["food_ordered"].tolist()

problem_items = (
    df[(df["taste_rating"] <= 2) & (df["price_rating"] >= 4)]
    .groupby("food_ordered")
    .size()
    .reset_index(name="Bad_Count")
)

bad_item = (
    problem_items.sort_values("Bad_Count", ascending=False)["food_ordered"].iloc[0]
    if not problem_items.empty
    else None
)

# --------------------------------------------------------
# ⭐ ADDED: Track each customer's BAD review history
# --------------------------------------------------------
bad_history = (
    df[df["taste_rating"] <= 2]
    .groupby("customer_id")
    .agg(
        Bad_Items=("food_ordered", list),
        Bad_Dates=("date", list)
    )
    .reset_index()
)

df = df.merge(bad_history, on="customer_id", how="left")
df["Bad_Items"] = df["Bad_Items"].apply(lambda x: [] if isinstance(x, float) else x)
df["Bad_Dates"] = df["Bad_Dates"].apply(lambda x: [] if isinstance(x, float) else x)

# -----------------------------
# 3. Build CUSTOMER HISTORY
# -----------------------------
customer_history = (
    df.groupby("customer_id")["food_ordered"]
      .apply(set)
      .reset_index()
      .rename(columns={"food_ordered": "Past_Items_Set"})
)

df = df.merge(customer_history, on="customer_id", how="left")
df["Past_Items_Set"] = df["Past_Items_Set"].apply(lambda x: set() if isinstance(x, float) else x)

# -----------------------------
# 4. Mark New vs Returning CUSTOMER
# -----------------------------
df["Is_New_Customer"] = df["Prev_Cluster"].isna()

# -----------------------------
# 5. TODAY items ordered
# -----------------------------
today_items = (
    df.groupby(["customer_id", "date"])["food_ordered"]
      .apply(set)
      .reset_index()
      .rename(columns={"food_ordered": "Today_Items_Set"})
)

df = df.merge(today_items, on=["customer_id", "date"], how="left")

# -----------------------------
# 6. Full Order History
# -----------------------------
df["Full_History_Set"] = df.apply(
    lambda row: row["Past_Items_Set"] | row["Today_Items_Set"],
    axis=1
)

# -----------------------------
# 7. RULE-A: Enhanced Top Sellers
# -----------------------------
def recommend_top_enhanced(items, is_new):
    recommendations = []

    if is_new:
        for item in top_items:
            if item not in items:
                recommendations.append(f"{item} (Top Suggestion – No discount)")
        return ", ".join(recommendations) if recommendations else None

    for item in top_items:
        if item not in items and random.random() < 0.5:
            recommendations.append(f"{item} (Top Suggestion – No discount)")

    return ", ".join(recommendations) if recommendations else None


df["Recommend_Top_Item"] = df.apply(
    lambda row: recommend_top_enhanced(row["Full_History_Set"], row["Is_New_Customer"]),
    axis=1
)

# -----------------------------
# 8. RULE-B Improve/Rework Logic (1 offer per month)
# -----------------------------
df["offer_month"] = pd.to_datetime(df["date"]).dt.to_period("M")
offer_tracker = {}

def recommend_rework(row):
    if bad_item is None:
        return None

    cust = row["customer_id"]
    month = row["offer_month"]
    key = (cust, month)

    if key not in offer_tracker:
        offer_tracker[key] = 0

    if offer_tracker[key] >= 1:
        return None

    rec = None

    if (row["food_ordered"] == bad_item) and (row["taste_rating"] <= 2):
        rec = f"{bad_item} (Try-Again Offer 15% Off)"

    elif row["Cluster"] in [1, 2, 3] and bad_item not in row["Full_History_Set"]:
        rec = f"{bad_item} (Intro Rework Offer 20% Off)"

    if rec:
        offer_tracker[key] += 1

    return rec

df["Recommend_Improve_Item"] = df.apply(recommend_rework, axis=1)

# ------------------------------------------------------------
# ⭐ PERSONALIZED MESSAGE TEMPLATES
# ------------------------------------------------------------
top_messages = [
    "You might enjoy → {} 😋",
    "A customer favorite → {} ⭐",
    "Perfect pick → {} ✨",
    "Matches your taste → {} 👌",
    "Highly recommended → {} 😍"
]

rework_messages = [
    "We improved this item — try again → {} 🔄",
    "You deserve a better taste — retry → {} 😊",
    "Special retry offer → {} 🎁",
    "This should taste better now → {} 🌟",
    "We fixed the issue — try this → {} 👍"
]

new_customer_messages = [
    "Great first choice → {} 😊",
    "A must-try for new customers → {} ⭐",
    "Perfect to start with → {} 🎉",
    "You’ll enjoy this → {} 😋",
    "A great starting pick → {} 🌟"
]

# --------------------------------------------------
# EVALUATION: Simulate New Customers from history
# --------------------------------------------------
eval_df = df.copy()

# Pick customers with enough history
eligible = eval_df.groupby("customer_id").filter(lambda x: len(x) >= 4)

# Randomly sample 20 real customers
sample_customers = eligible["customer_id"].drop_duplicates().sample(20, random_state=42)

evaluation_results = []

for cust in sample_customers:
    history = eval_df[eval_df["customer_id"] == cust].sort_values("date")

    first_visit = history.iloc[0]
    later_visits = history.iloc[1:]

    t = first_visit["taste_rating"]
    p = first_visit["price_rating"]

    new_features = pd.DataFrame([[t, p]], columns=["Avg_Taste", "Avg_Price"])
    new_features_scaled = scaler.transform(new_features)
    predicted_cluster = kmeans.predict(new_features_scaled)[0]

    favorite = first_visit["food_ordered"]

    cluster_history = (
       eval_df[eval_df["Cluster"] == predicted_cluster]
      .groupby("food_ordered")
      .size()
      .sort_values(ascending=False)
    )
    recs=[]
    recs.append(favorite)
    for item in cluster_history.index:
        if item not in recs:
           recs.append(item)
        if len(recs) == 3:
           break
    
    recs = set(recs)


    
    future_items = set(later_visits["food_ordered"])

    hit_rate = len(recs & future_items) / max(len(recs), 1)

    evaluation_results.append({
        "customer_id": cust,
        "hit_rate": hit_rate
    })

# -----------------------------
# 📊 DISPLAY EVALUATION RESULTS
# -----------------------------
eval_table = pd.DataFrame(evaluation_results)

print("\n🧪 New Customer Recommendation Evaluation")
print("----------------------------------")
print(eval_table.to_string(index=False))

overall_hit_rate = eval_table["hit_rate"].mean()
print(f"\n📈 Overall Hit Rate: {overall_hit_rate:.2f}")

def engine_recommend(customer_name, taste=None, price=None, category=None):
    df["name_lower"] = df["customer_name"].str.lower()
    matches = df[df["name_lower"].str.contains(customer_name.lower(), na=False)]

    # ---------------- NEW CUSTOMER ----------------
    if matches.empty:
        # ---------- Safety Conversion ----------
        try:
           taste = float(taste)
           price = float(price)
        except:
           raise ValueError("Taste and Price must be selected before getting recommendations.")

        new_features = pd.DataFrame([[taste, price]], columns=["Avg_Taste", "Avg_Price"])

        new_scaled = scaler.transform(new_features)
        new_cluster = kmeans.predict(new_scaled)[0]

        menu = cluster_menu.get(new_cluster, MENU_ITEMS)
        base_recos = random.sample(menu, min(3, len(menu)))

        cat_reco = category_items.get(category, ["Cold Coffee"])[0]

        return {
            "type": "new",
            "cluster": int(new_cluster),
            "recommendations": base_recos,
            "category_pick": cat_reco
        }

    # ---------------- EXISTING CUSTOMER ----------------
    customer = matches.iloc[-1]
    history = df[df["customer_id"] == customer["customer_id"]]

    favorite = history["food_ordered"].value_counts().idxmax()
    cluster = customer["Cluster"]

    cluster_items = (
        df[df["Cluster"] == cluster]["food_ordered"]
        .value_counts()
        .index.tolist()
    )

    recs = [favorite]
    for item in cluster_items:
        if item not in recs:
            recs.append(item)
        if len(recs) == 3:
            break

    return {
        "type": "existing",
        "customer_id": customer["customer_id"],
        "recommendations": recs
    }


# -----------------------------
# 9. Ask user for customer name
# -----------------------------
# name_input = input("Enter Customer Name: ").strip().lower()
# df["name_lower"] = df["customer_name"].str.lower()

# partial_matches = df[df["name_lower"].str.contains(name_input, na=False)]

# # ==============================================================================================
# # NEW CUSTOMER LOGIC
# # ==============================================================================================
# if partial_matches.empty:
#     print("\n🆕 Customer not found → Treating as NEW CUSTOMER")

#     t = int(input("Enter Taste Rating (1-5): "))
#     p = int(input("Enter Price Rating (1-5): "))

#     print("\nChoose Taste Preference Category:")
#     print("[spicy, sweet, cheesy, healthy, neutral]")
#     cat = input("Enter category: ").strip().lower()

#     # def assign_cluster(taste, price):
#     #     return 7 if taste >= 4 and price >= 4 else 3
#     # --- ML-based cluster assignment for new customer ---
#     new_features = pd.DataFrame([[t, p]], columns=["Avg_Taste", "Avg_Price"])
#     new_features_scaled = scaler.transform(new_features)
#     new_cluster = kmeans.predict(new_features_scaled)[0]

#     print(f"\nAssigned ML Cluster: {new_cluster}")





#     menu = cluster_menu.get(new_cluster, MENU_ITEMS)
#     base_recos = random.sample(menu, min(3, len(menu)))

#     cat_reco = category_items.get(cat, ["Cold Coffee"])[0]

#     top_item = base_recos[0]
#     combo = None
#     for key in combo_map:
#         if key.lower() in top_item.lower():
#             combo = combo_map[key]
#             break

#     print("\n🎯 FINAL RECOMMENDATIONS FOR NEW CUSTOMER:\n")

#     print("⭐ Based on Cluster:")
#     for item in base_recos:
#         msg = random.choice(new_customer_messages)
#         print("👉", msg.format(item))

#     print("\n🍽 Based on Taste Preference Category:")
#     msg = random.choice(new_customer_messages)
#     print("👉", msg.format(cat_reco))

#     print("\n🥤 Suggested Combo:")
#     print("👉", combo if combo else "No combo available")

# # ==============================================================================================
# # EXISTING CUSTOMER LOGIC
# # ==============================================================================================
# else:
#     unique_customers = partial_matches[["customer_id", "customer_name"]].drop_duplicates().reset_index(drop=True)

#     if len(unique_customers) > 1:
#         print("\nMultiple customers found with this name:")
#         for i, row in unique_customers.iterrows():
#             print(f"{i+1}. {row['customer_name']} ({row['customer_id']})")
#         choice = int(input("\nSelect which customer → ")) - 1
#         selected_id = unique_customers.iloc[choice]["customer_id"]
#     else:
#         selected_id = unique_customers.iloc[0]["customer_id"]

#     customer_df = df[df["customer_id"] == selected_id].sort_values("date")
#     latest = customer_df.iloc[-1]

#     print("\n===================================================")
#     print(f"📌 CUSTOMER SUMMARY → {latest['customer_name']} ({selected_id})")

#     # ------------------------------------------------------------
#     # ⭐ SHOW PAST VISIT DATE
#     # ------------------------------------------------------------
#     if len(customer_df) > 1:
#         prev_date = customer_df.iloc[-2]["date"]
#         print(f"🕒 Previous Visit Date → {prev_date}")

#     # ------------------------------------------------------------
#     # ⭐ SHOW BADLY RATED ITEM & DATE
#     # ------------------------------------------------------------
#     if latest["Bad_Items"]:
#         print("\n⚠ Past Bad Ratings Detected:")
#         for item, d in zip(latest["Bad_Items"], latest["Bad_Dates"]):
#             print(f"❌ {item} (Rated badly on {d})")

#     print("\n🎯 Personalized Recommendations:")
#     print("--------------------------------------------------")

#     # ⭐ Personalized Top Seller
#     if latest["Recommend_Top_Item"]:
#         items = [i.split("(")[0].strip() for i in latest["Recommend_Top_Item"].split(",")]
#         for item in items:
#             msg = random.choice(top_messages)
#             print("⭐", msg.format(item))

#     # ⭐ Personalized Rework Suggestion
#     if latest["Recommend_Improve_Item"]:
#         item = latest["Recommend_Improve_Item"].split("(")[0].strip()
#         msg = random.choice(rework_messages)
#         print("💡", msg.format(item))

#         # ⭐ EXTRA: Tell WHY it is recommended
#         print("   ↳ Because you rated it poorly earlier.")


# ================================
# STREAMLIT API FUNCTIONS
# ================================
def get_recommendations(name, taste=None, price=None, category=None):
    return engine_recommend(name, taste, price, category)

# def get_recommendations(customer_name):
#     global df

#     customer = df[df["customer_name"].str.lower() == customer_name.lower()]

#     # If new customer
#     if customer.empty:
#         return {
#             "type": "new",
#             "recommendations": random.sample(MENU_ITEMS, 3)
#         }

#     # Existing customer
#     history = customer["food_ordered"]
#     favorite = history.value_counts().idxmax()

#     similar_cluster = customer.iloc[-1]["Cluster"]
#     cluster_items = (
#         df[df["Cluster"] == similar_cluster]["food_ordered"]
#         .value_counts()
#         .index.tolist()
#     )

#     recs = [favorite]
#     for item in cluster_items:
#         if item not in recs:
#             recs.append(item)
#         if len(recs) == 3:
#             break

#     return {
#         "type": "existing",
#         "recommendations": recs
#     }


def evaluate_system():
    return overall_hit_rate
