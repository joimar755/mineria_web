from typing import Optional, Union
from fastapi import Form
from pydantic import BaseModel

class users(BaseModel):
    username:  str | None = None
    password:  str | None = None
    
    class Config:
        from_attributes = True

class Login(BaseModel):
    username:  str 
    password:  str 
    
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    id: Optional[int] = None
    #username: Union[str, None] = None