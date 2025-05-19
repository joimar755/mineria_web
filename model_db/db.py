import datetime
from sqlalchemy import TIMESTAMP, DateTime, Float, Integer, String, Table, Column, text, true, ForeignKey
from sqlalchemy.orm import relationship
from config.db import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(255))
    vendedor = Column(String(255))
    modelo = Column(String(100))  # Linear, RandomForest, Logistic, etc.
    tipo = Column(String(50))     # regresion / clasificacion
    valor = Column(Float)
    fecha = Column(DateTime, default=datetime.utcnow) 
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    
class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)