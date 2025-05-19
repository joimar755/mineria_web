from typing import Optional
from pydantic import BaseModel

class vhBase(BaseModel):
    nombre: str
    vendedor: str
    modelo: str
    tipo: str
    valor: str
   

class vhcreate(vhBase):
 pass

class vh(vhBase):
    id:int 
    user_id: Optional[int]

    class Config:
        from_attributes = True