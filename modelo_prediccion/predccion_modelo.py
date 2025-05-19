from typing import Optional
from pydantic import BaseModel

class vhBase(BaseModel):
    nombre: str
    vendedor: str
    modelo: str
    tipo: str
    valor: float
    user_id: int  # ✅ Obligatorio si lo usa Prediction
    fecha: Optional[str] = None  # opcional, si lo necesitas

class vhcreate(vhBase):
    pass

class vh(vhBase):
    id: int

    class Config:
        orm_mode = True
