from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os 
from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL_SENSOR")
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://u3iur9ccnfeskrys:HJLagOR3ISDZxNUB8rhS@bcub4ompt88fqjbdso8m-mysql.services.clever-cloud.com:3306/bcub4ompt88fqjbdso8m"
engine = create_engine(
    DATABASE_URL
    #SQLALCHEMY_DATABASE_URL,
     pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
#Base.metadata.create_all(bind=engine, checkfirst=True)

def get_db():
    db = SessionLocal() 
    try:
        yield db
    finally:
        db.close()



