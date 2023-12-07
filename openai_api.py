# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Body, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

# Custom
import os
import json
from typing_extensions import Annotated
from dotenv import load_dotenv
from starlette import status
from fastapi.security import APIKeyHeader
from transformers import BertTokenizer, BertModel
from fastapi.responses import JSONResponse

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

access_token = os.environ.get("ACCESS_TOKEN", "")

api_key_header = APIKeyHeader(name="Authorization")

def get_current_user_deps(token: Annotated[str, Depends(api_key_header)]):
    if token != access_token:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/v1/models", response_model=ModelList)
async def list_models(_: Annotated[str, Depends(api_key_header)]):
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest = None):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")
    #return JSONResponse(content={"model": request.model, "choices":[choice_data], "object":"chat.completion"}, media_type="application/json")


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_glm_embedding(text, device="cuda"):
    global model_embedding, tokenizer_embedding
    
    # inputs = tokenizer([text], return_tensors="pt").to(device)
    encoded_input = tokenizer_embedding(text, padding=True, truncation=True, return_tensors="pt").to(device)
    # resp = model.transformer(**inputs, output_hidden_states=True)
    # y = resp.last_hidden_state
    # y_mean = torch.mean(y, dim=0, keepdim=True)
    # result = y_mean.cpu().detach().numpy()
    # return result
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print("Sentence embeddings:", flush=True)
    print(sentence_embeddings, flush=True)
    return sentence_embeddings
  
  
@app.post("/v1/embeddings")
# async def create_embeddings(_: Annotated[str, Depends(api_key_header)], text: Annotated[str, Body(embed=True)] = None):
async def create_embeddings(
    # _: Annotated[str, Depends(api_key_header)], 
    # text: Annotated[str, Body(embed=True)] = None
    input: List[str] = Body(..., embed=True)
):
    embedding_obj = get_glm_embedding(input)
    embedding_list = embedding_obj.tolist()
    return_dict = {"data": {"embedding": [embedding_list]}}
    json_dict = json.dumps(return_dict)
    # print(f"create_embeddings json_str={json_str}", flush=True)
    return json_dict


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True).cuda()
    tokenizer_embedding = BertTokenizer.from_pretrained("./text2vec-large-chinese/vocab.txt", trust_remote_code=True,local_files_only=True)
    model_embedding = BertModel.from_pretrained("./text2vec-large-chinese/pytorch_model.bin",config='./text2vec-large-chinese/config.json', trust_remote_code=True, local_files_only=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    #model.eval()

    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)), workers=1)
