import pandas as pd

def load_data():
    input_path = '/home/katya/airflow/data/student-mat.csv'
    output_path = '/home/katya/airflow/data/student_raw.pkl'

    df = pd.read_csv(input_path, sep = ';')
    print(f'strings were load: {len(df)}')

    df.to_pickle(output_path)
    return len(df)
if __name__ == '__main__':
    load_data() 
