from sqlalchemy.orm import Session
from modelo_prediccion import predccion_modelo
from model_db.db import Prediction

def guardar_prediccion(pred_data: predccion_modelo.vhBase, db: Session):
    pred = Prediction(
        nombre=pred_data.nombre,
        vendedor=pred_data.vendedor,
        modelo=pred_data.modelo,
        tipo=pred_data.tipo,
        valor=pred_data.valor
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred
