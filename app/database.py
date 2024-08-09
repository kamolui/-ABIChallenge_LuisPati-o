"""from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the SQLAlchemy engine (replace with your actual database URL)
DATABASE_URL = "postgresql://user:password@localhost/dbname"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Prediction model
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String, index=True)

# Create the database tables
Base.metadata.create_all(bind=engine)

def store_prediction(prediction):
    db = SessionLocal()
    db_prediction = Prediction(prediction=str(prediction))
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    db.close()
"""