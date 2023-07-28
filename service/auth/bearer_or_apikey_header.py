from typing import Optional
from fastapi.security import OAuth2PasswordBearer
from starlette.requests import Request

class OAuth2PasswordBearerOrApiKeyHeader(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        apikey = request.headers.get("api-key")
        if apikey is not None:
            return apikey
        return await super().__call__(request)
