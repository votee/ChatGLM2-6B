from service.auth.jwt_utils import validate_access_token
from service.auth.schema import (
    AppAuthError,
    AppTokenPayload,
    AppUser,
)
from datahandler.mongodb_handler import MongoHelper

def find_user(
    token: str,
) -> AppUser:
    token_data_or_error = validate_access_token(token=token)
    if isinstance(token_data_or_error, AppTokenPayload):
        token_data: AppTokenPayload = token_data_or_error
        db = MongoHelper()
        user = db.get_user(email=token_data.sub)
        if user == None:
            raise AppAuthError.USER_NOT_EXISTED
        return user
    else:
        error: AppAuthError = token_data_or_error
        raise error