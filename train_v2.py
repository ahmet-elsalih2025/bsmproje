from sklearn.datasets import fetch_openml  # MNIST veri setini indirir
from sklearn.neural_network import MLPClassifier # burda gizli katmanlar oluşur 
from sklearn.model_selection import train_test_split # veri setini test ve öğrenme olarak ayırır
import joblib # yapay zeka öğrendiğini unutmamak için
import numpy as np # hızlı matematik işlemler için

def train():
    print("Loading MNIST data (this might take a while)...")
    # Fetch MNIST data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Normalize
    X = X / 255.0 

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print("Training MLPClassifier...")
    # MLP with 1 hidden layer of 100 neurons should be enough for ~97% accuracy
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, alpha=1e-4,  
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)

    clf.fit(X_train, y_train)

    print("Training done. Evaluating...")
    score = clf.score(X_test, y_test)
    print(f"Test Accuracy: {score}")

    # Save
    joblib.dump(clf, "mnist_model.pkl")
    print("Model saved to mnist_model.pkl")

if __name__ == "__main__":
    train()
