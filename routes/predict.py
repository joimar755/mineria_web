from datetime import timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from config.db import  SessionLocal, get_db
from passlib.context import CryptContext
from modelo_prediccion import oauth
from modelo_prediccion.oauth import get_current_user
from model_db.db import  Users
from sqlalchemy.orm import Session
from model_db import db
from modelo_prediccion import predccion_modelo
from model_db.pred import guardar_prediccion 
from modelo_prediccion.vendedor_mineria import Item

from modelo_prediccion.predccion_modelo import vhBase
from modelo_prediccion.prediccion_user import Login, Token, users
from modelo_prediccion.token import create_access_token
import json

import pandas as pd
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

df = pd.read_csv("electrodomesticos_mercadolibre.csv")
df = df[['Nombre', 'Vendedor', 'Precio']].dropna()
precio_promedio = df['Precio'].mean()
df['Precio_Alto'] = (df['Precio'] > precio_promedio).astype(int)

# Preparadores
tfidf = TfidfVectorizer(max_features=100)
tfidf.fit(df['Nombre'])
le = LabelEncoder()
le.fit(df['Vendedor'])

# Entrenamiento previo
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error, mean_absolute_error

X_nombre = tfidf.transform(df['Nombre'])
X_vendedor = le.transform(df['Vendedor'])
X = hstack([X_nombre, np.array(X_vendedor).reshape(-1, 1)])
y_reg = df['Precio']
y_clf = df['Precio_Alto']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2)

# Modelos
lr = LinearRegression().fit(X_train_r, y_train_r)
rf_reg = RandomForestRegressor().fit(X_train_r, y_train_r)
log_clf = LogisticRegression(max_iter=1000).fit(X_train_c, y_train_c)
rf_clf = RandomForestClassifier().fit(X_train_c, y_train_c)


VH = APIRouter()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@VH.post("/predict/clasificacion")
def predict_clasificacion(item: Item,
                          db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user)):
    X_nombre = tfidf.transform([item.nombre])
    X_vendedor = le.transform([item.vendedor])
    X_input = hstack([X_nombre, [[X_vendedor[0]]]])

    pred_log = int(log_clf.predict(X_input)[0])
    pred_rf = int(rf_clf.predict(X_input)[0])

    db = SessionLocal()

    # Usamos vhBase para construir la predicción y guardarla
    pred1 = vhBase(
        nombre=item.nombre,
        vendedor=item.vendedor,
        modelo="Logistic Regression",
        tipo="clasificacion",
        valor=pred_log,
        user_id=current_user.id 
    )
    guardar_prediccion(pred1, db)

    pred2 = vhBase(
        nombre=item.nombre,
        vendedor=item.vendedor,
        modelo="Random Forest Classifier",
        tipo="clasificacion",
        valor=pred_rf,
        user_id=current_user.id 
    )
    guardar_prediccion(pred2, db)

    db.close()

    return {
        "logistic_regression": pred_log,
        "random_forest_classifier": pred_rf
    }
@VH.post("/predict/regresion")
def predict_regresion(
    item: Item,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    X_nombre = tfidf.transform([item.nombre])
    X_vendedor = le.transform([item.vendedor])
    X_input = hstack([X_nombre, [[X_vendedor[0]]]])

    pred_lr = lr.predict(X_input)[0]
    pred_rf = rf_reg.predict(X_input)[0]

    pred1 = vhBase(
        nombre=item.nombre,
        vendedor=item.vendedor,
        modelo="Linear Regression",
        tipo="regresion",
        valor=pred_lr,
        user_id=current_user.id  # ✅ ahora lo pasas
    )
    guardar_prediccion(pred1, db)

    pred2 = vhBase(
        nombre=item.nombre,
        vendedor=item.vendedor,
        modelo="Random Forest Regressor",
        tipo="regresion",
        valor=pred_rf,
        user_id=current_user.id  # ✅ aquí también
    )
    guardar_prediccion(pred2, db)

    return {
        "linear_regression": pred_lr,
        "random_forest_regressor": pred_rf
    }

@VH.get("/metrics")
def get_metrics():
    # Regresión
    pred_lr = lr.predict(X_test_r)
    pred_rf = rf_reg.predict(X_test_r)

    metrics_reg = {
        "Linear Regression": {
            "R2": r2_score(y_test_r, pred_lr),
            "MAE": mean_absolute_error(y_test_r, pred_lr),
            "RMSE": np.sqrt(mean_squared_error(y_test_r, pred_lr))
        },
        "Random Forest Regressor": {
            "R2": r2_score(y_test_r, pred_rf),
            "MAE": mean_absolute_error(y_test_r, pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_test_r, pred_rf))
        }
    }

    # Clasificación
    pred_log = log_clf.predict(X_test_c)
    pred_rf_clf = rf_clf.predict(X_test_c)

    metrics_clf = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test_c, pred_log),
            "F1 Score": f1_score(y_test_c, pred_log)
        },
        "Random Forest Classifier": {
            "Accuracy": accuracy_score(y_test_c, pred_rf_clf),
            "F1 Score": f1_score(y_test_c, pred_rf_clf)
        }
    }

    return {
        "regresion": metrics_reg,
        "clasificacion": metrics_clf
    }




@VH.get("/profile/{id}")
def index(id: int, db: Session = Depends(get_db),current_user: int = Depends(oauth.get_current_user)):
    vh_query = db.query(Users).filter(Users.id == id)
    vh = vh_query.first()
    if vh == None:
        raise HTTPException(status_code=404, detail="user not found")
    if vh.id != current_user.id:
        raise HTTPException(status_code=401, detail="Unauthorized to users") 
    
    return {"user":vh}


@VH.post("/usuario")
def get_user(user: users,db: Session = Depends(get_db)):
    existe = db.query(Users).filter(Users.username == user.username).first()
    if existe:
        return JSONResponse("usuario ya se encuentra en uso")
    if existe is None:
            hashed_password = pwd_context.hash(user.password)
            user.password = hashed_password
            db_item = Users(**user.model_dump())
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
    else:
        raise HTTPException(
            status_code=404, detail="product with this name already exists "
         )
    #vh_query = db.query(Users).filter(Users.id == db_item.id).first()
    return  db_item


@VH.post("/usuario/login")
def get_user(user_credentials:OAuth2PasswordRequestForm=Depends(),db: Session = Depends(get_db)):
    user = db.query(Users).filter(Users.username == user_credentials.username).first()

    if not user or not pwd_context.verify(user_credentials.password, user.password):
        return JSONResponse("Incorrect username or password")
        raise HTTPException(409, "Incorrect username or password")

    access_token =  create_access_token(data={"user_id": user.id})
    token = {"access_token": access_token, "username":user.username, "userID":user.id, "token_type": "bearer"}
    return JSONResponse(token)



# raise HTTPException(status_code=200, detail="login")