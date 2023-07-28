import os
from pymongo import MongoClient
from service.auth.schema import AppUserCredentials
from pymongo.collection import Collection
from typing import Any

MONGO_URL = os.environ.get("MONGO_URL", None)
client: MongoClient = MongoClient(MONGO_URL)

MONGO_DATABASE_USERS_COLLECTION = "users"

class MongoHelper:
    def __init__(self):
        self.db = client["cugpt"]
        
    def get_user(self, email: str) -> AppUserCredentials:
        collection: Collection = self.db[MONGO_DATABASE_USERS_COLLECTION]
        condition = {"email": email}
        item: dict[str, Any] = collection.find_one(condition)
        if item == None:
            raise Exception("User not found")
        
        item["id"] = str(item["_id"])
        item.pop("_id")
        return AppUserCredentials.parse_obj(item)