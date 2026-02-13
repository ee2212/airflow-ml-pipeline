from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.insert(0, '/home/katya/airflow/dags/scripts')

from load_students import load_data
from preprocess_student import preprocess_data
from train_student import train_model

with DAG(
    'student_success_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description='Pipeline for predicting student success'
) as dag:
    
    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    t1 >> t2 >> t3
