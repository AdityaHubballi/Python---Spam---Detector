import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "label": [
        "spam","ham","spam","ham","spam","ham","spam","ham",
        "spam","ham","spam","ham","spam","ham"
    ],
    "message": [
        "Win money now",
        "Hi how are you",
        "Claim your free prize",
        "Let us meet tomorrow",
        "Congratulations you won a lottery",
        "Are you coming today",
        "Free offer just for you",
        "See you at college",
        "Earn cash fast now",
        "Can we talk later",
        "Free recharge offer",
        "Please call me",
        "Winner claim reward",
        "Good morning friend"
    ]
}

df = pd.DataFrame(data)

# Convert labels to numbers
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.5, random_state=42)

# Convert text to numeric form
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Live test
user_message = input("Enter a message to check: ")
user_vec = vectorizer.transform([user_message])
result = model.predict(user_vec)

if result[0] == 1:
    print("🚨 Spam Message Detected")
else:
    print("✅ Not a Spam Message")
