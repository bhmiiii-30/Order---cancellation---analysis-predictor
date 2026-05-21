import sqlite3
import pandas as pd
import pickle

# ---------- SQL DATA ----------
sql_script = """
CREATE TABLE orders_new (
    order_id TEXT PRIMARY KEY,
    order_date DATE,
    city TEXT,
    state TEXT,
    locality TEXT,
    pincode INTEGER,
    sales_channel TEXT,
    customer_segment TEXT,
    product_category TEXT,
    payment_method TEXT,
    order_amount REAL,
    potential_revenue REAL,
    actual_revenue REAL,
    revenue_loss REAL,
    discount_percent INTEGER,
    promised_delivery_days INTEGER,
    actual_delivery_days INTEGER,
    delivery_delay_days INTEGER,
    order_status TEXT,
    cancelled_flag INTEGER,
    cancellation_reason TEXT
);

INSERT INTO orders_new VALUES
('1','2026-02-03','Hyd','TS','A',500001,'App','New','Fashion','COD',4000,4000,0,4000,20,3,NULL,NULL,'Cancelled',1,'Address'),
('2','2026-03-10','Hyd','TS','B',500002,'App','Premium','Books','UPI',1500,1500,1500,0,10,2,2,0,'Delivered',0,NULL),
('3','2026-04-12','Hyd','TS','C',500003,'Website','Regular','Electronics','COD',10000,10000,0,10000,25,5,NULL,NULL,'Cancelled',1,'Stock')
"""

# ---------- CREATE DB ----------
conn = sqlite3.connect(":memory:")
conn.executescript(sql_script)

df = pd.read_sql_query("SELECT * FROM orders_new", conn)

# ---------- PREPROCESS ----------
df["order_date"] = pd.to_datetime(df["order_date"])
df["cancelled_flag"] = df["cancelled_flag"].astype(int)

ml_df = df.drop(columns=["order_id", "cancellation_reason", "city", "state"])

ml_df["actual_delivery_days"] = ml_df["actual_delivery_days"].fillna(-1)
ml_df["delivery_delay_days"] = ml_df["delivery_delay_days"].fillna(-1)

ml_df["order_month"] = ml_df["order_date"].dt.month
ml_df["order_day"] = ml_df["order_date"].dt.day
ml_df = ml_df.drop(columns=["order_date"])

# ---------- ENCODING ----------
ml_df = pd.get_dummies(ml_df, drop_first=True)

# ---------- SPLIT ----------
from sklearn.model_selection import train_test_split

X = ml_df.drop("cancelled_flag", axis=1)
y = ml_df["cancelled_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------- MODEL ----------
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# ---------- SAVE ----------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("✅ Model saved successfully")