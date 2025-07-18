# ✅ Test message to confirm script start
print("✅ Test message: Script started")

# 📦 Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 📥 Load dataset
df = pd.read_csv("data/creditcard.csv")
print("✅ CSV Loaded! Shape:", df.shape)

# 🎯 Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🤖 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 Predict
predictions = model.predict(X_test)

# 🧠 Evaluate model
print("\n🔍 Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\n📄 Classification Report:\n", classification_report(y_test, predictions))

# ✅ Final status
print("\n✅ main.py script executed successfully 🎉")

# ⏸️ Keep window open so user can see result
input("🔚 Press Enter to close the script...")



