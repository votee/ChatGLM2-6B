import os
import traceback
from pydantic import ValidationError
from datetime import datetime
from jose import jwt, JWTError
from service.auth.schema import AppAuthError, AppTokenPayload
from dotenv import load_dotenv

load_dotenv()

ALGORITHM = "HS256"
JWT_SECRET_KEY = os.environ['JWT_SECRET_KEY'] # should be kept secret
JWT_REFRESH_SECRET_KEY = os.environ['JWT_REFRESH_SECRET_KEY'] # should be kept secret

def validate_access_token(token: str) -> AppTokenPayload:
    return _validate_token(
        token=token,
        secret_key=JWT_SECRET_KEY,
    )


def validate_refresh_token(token: str) -> AppTokenPayload:
    return _validate_token(
        token=token,
        secret_key=JWT_REFRESH_SECRET_KEY,
    )


def _validate_token(token: str, secret_key: str) -> AppTokenPayload:
    try:
        payload = jwt.decode(
            token=token,
            key=secret_key,
            algorithms=[ALGORITHM],
        )
        token_data = AppTokenPayload(**payload)
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise AppAuthError.TOKEN_EXPIRED

        return token_data
    except(JWTError, ValidationError):
        traceback.print_exc()
        raise AppAuthError.CANNOT_VALIDATE_CREDENTIALS