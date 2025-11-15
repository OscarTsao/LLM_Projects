FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install PostgreSQL driver for MLflow backend store
RUN pip install --no-cache-dir psycopg2-binary

# Expose MLflow server port
EXPOSE 5000
