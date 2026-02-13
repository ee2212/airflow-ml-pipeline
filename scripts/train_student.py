import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # paths
    data_path = '/home/katya/airflow/data/student_processed.pkl'
    model_path = '/home/katya/airflow/models/student_model.pkl'
    metrics_path = '/home/katya/airflow/metrics/student_metrics.json'
    
    # load preprocessed data
    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, scaler, label_encoders, features = pickle.load(f)
    
    # create and learn model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # prediction and accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # save model
    with open(model_path, 'wb') as f:
        pickle.dump((model, scaler, label_encoders, features), f)
    
    # save metrics
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForest',
        'test_samples': len(y_test)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model was learned. Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__" :
    train_model()
