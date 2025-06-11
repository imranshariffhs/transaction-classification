import pandas as pd
from sqlalchemy import create_engine

db_url = "postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db"
engine = create_engine(db_url)

df = pd.read_sql("SELECT * FROM temp_transactions", engine)
df.to_csv("output.csv", index=False)
