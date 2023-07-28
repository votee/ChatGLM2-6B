from pydantic import BaseModel, Field
from enum import Enum

class AppUser(BaseModel):
    id: str
    email: str = Field(..., description="user email")
    name: str

class AppUserCredentials(AppUser):
    password: str    

class AppTokenPayload(BaseModel):
    sub: str
    exp: int

class AppAuthError(Enum):
    UNKNOWN = 1
    EMAIL_EXISTED = 2
    EMAIL_NOT_EXISTED = 3
    PASSWORD_INCORRECT = 4
    TOKEN_EXPIRED = 5
    CANNOT_VALIDATE_CREDENTIALS = 6
    USER_NOT_EXISTED = 7