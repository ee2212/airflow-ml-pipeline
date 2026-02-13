import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    input_path = '/home/katya/airflow/data/student_raw.pkl'
    output_path = '/home/katya/airflow/data/student_processed.pkl'
    
    df = pd.read_pickle(input_path)
    
    # target -- pass
    df['pass'] = (df['G3'] >= 10).astype(int)
    
    # (all abovе G1, G2, G3)
    features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                'paid', 'activities', 'nursery', 'higher', 'internet',
                'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                'health', 'absences']
    
    X = df[features]
    y = df['pass']
    
    # categorial
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # save
    with open(output_path, 'wb') as f:
        pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, features), f)
    
    print(f"Preproceed success. Train df: {len(X_train)} strings")

if __name__ == "__main__":
    preprocess_data()
