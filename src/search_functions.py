import json
from dotenv import load_dotenv 
import os 
import tiktoken
import openai 
from openai import OpenAI
from groq import Groq
import discord
from datetime import timezone
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import ChromaCloudSpladeEmbeddingFunction
from chromadb.execution.expression.operator import K
from chromadb import Search, K, Knn, Rrf, Schema, SparseVectorIndexConfig, K
import ast
import time
import asyncio
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
chromadb_api_key = os.getenv("CHROMA_API_KEY")
chromadb_tenant= os.getenv("CHROMADB_TENANT")
chromadb_database= os.getenv("CHROMADB_DATABASE")
chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
general_channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))
intro_channel_id = int(os.getenv('DISCORD_INTRO_CHANNEL_ID'))
notes_channel_id = int(os.getenv('DISCORD_NOTES_CHANNEL_ID'))
groq_api_key = os.getenv('groq_api_key')

os.environ['CHROMA_API_KEY'] = chromadb_api_key

channels = [general_channel_id, intro_channel_id, notes_channel_id]

general_channel_starting_chunk_index = 163

prompt_name = f"{os.path.join(os.getcwd(), "prompts/search_prompt.txt")}"

server_users = {
    "ohsheetklaus": "Steven",
    "ShortSideburns": "Vihan",
    "1kdev": "1k",
    "pope": "Pope",
    "search": "BOT",
    "m1keromano": "Mike Romano",
    "RC": "Rita",
    "yourboymaleek": "Malik"
}

client = discord.Client(intents=discord.Intents.default())
chroma_client = chromadb.CloudClient(
  api_key= chromadb_api_key,
  tenant= chromadb_tenant, 
  database= chromadb_database
)
schema = Schema()
openai_client = OpenAI(
    api_key = openai_api_key
)


sparse_ef = ChromaCloudSpladeEmbeddingFunction()
schema.create_index(
    config=SparseVectorIndexConfig(
        source_key=K.DOCUMENT,
        embedding_function=sparse_ef
    ),
    key="sparse_embedding"
)

collection_anduril = chroma_client.get_or_create_collection(
    name=chroma_collection_name,
    schema=schema
)

def run_hybrid_search(collection, query, limit=30):
    """Combined semantic and keyword search using RRF."""
    ranker = Rrf(
        ranks=[
            Knn(query=query, return_rank=True),
            Knn(query=query, key="sparse_embedding", return_rank=True)
        ],
        weights=[0.5, 0.5],
        k=100
    )
    search = (Search()
              .rank(ranker)
              .limit(limit)
              .select(K.DOCUMENT, K.SCORE))
    results = collection.search(search)
    return results

async def retrieve_collection(collection_name: str, schema: Schema, client: chromadb.CloudClient): 
    collection = await client.get_or_create_collection(name=collection_name,schema=schema)
    return collection

def retrieve_prompt(prompt_name: str) -> str: 
    prompt = open(prompt_name, 'r', encoding='utf-8').read()
    return prompt

def answer_query(system_prompt: str) -> str: 
    response = openai_client.responses.create(
        model="gpt-5.4-nano",
        input=system_prompt 
    )
    response_text = response.output_text
    return response_text

def groq_answer(system_prompt: str) -> str: 
   
    client = Groq(api_key = groq_api_key)
    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      
    ],
    temperature=0.2,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None)

    groq_response = completion.choices[0].message.content or ""
    return groq_response
        
formatted_messages = []

def chunk_documents(messages: list, chunk_size: int = 10, overlap: int = 3):
    step = chunk_size - overlap

    for chunk_start in range(0, len(messages), step):
        chunk = messages[chunk_start:chunk_start + chunk_size]
        print(f'chunk: {chunk}')

        if not chunk:
            continue

        chunk_end = chunk_start + len(chunk) - 1

        ids = [
            f"chunk_{chunk_start}_{chunk_end}"
        ]

        documents = [
            "\n".join(
                f'sender_name: {msg["sender_name"]}\nmessage: {msg["text"]}'
                for msg in chunk
            )
        ]

        metadatas = [
            {
                "messages_metadata": json.dumps([
                    {
                        "sender": msg["sender_name"],
                        "message_id": str(chunk_start + i)
                    }
                    for i, msg in enumerate(chunk)
                ])
            }
        ]


        collection_anduril.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Uploaded anduril chunk {chunk_start} to {chunk_end}")


@client.event
async def on_rate_limit(payload):
    print(f'Rate limit hit -> bucket: {payload.bucket}')

# @client.event
# async def on_ready():
#     for channel in channels: 
#         channel = client.get_channel(channel)
#         print(f'channel: {channel}')
        
#         users = set()
#         async for msg in channel.history(limit=None):
#             users.add((msg.author.id, msg.author.display_name))
        
#         print(f"Got {len(users)} unique users")
#         print(f'Users: {users}')
        
#     await client.close()

@client.event
async def on_ready():
    all_messages = []
    for channel in channels: 
        channel = client.get_channel(channel)
        print(f'channel: {channel}')
        messages = [
            {
                "sender_name": server_users.get(msg.author.display_name, msg.author.display_name), 
                "text": msg.content, 
                "date": msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            async for msg in channel.history(limit = None)
        ]
        print(f"Got {len(messages)} messages")
        all_messages.extend(messages)

        with open('discord_messages_all', 'w', encoding='utf-8') as file: 
            file.write(str(all_messages))
    await client.close()

def retrieve_messages(messages_path: str = 'discord_messages'):
    messages = ast.literal_eval(open(messages_path, 'r', encoding='utf-8').read())
    senders = set([message.get('sender_name', '') for message in messages])
    print(len(senders))
    print(senders)

def get_token_count(text: str) -> int: 
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    return len(tokens)

def full_pipeline_inference(prompt_name: str, chroma_collection: str, user_query: str, limit: int = 30) -> str: 
    responses = run_hybrid_search(collection = chroma_collection, query = user_query, limit = limit)
    documents = responses.get('documents', [])[:10]
    system_prompt = retrieve_prompt(prompt_name).format(USER_QUERY = user_query, DOCS = documents)
    llm_response = answer_query(system_prompt = system_prompt)
    return llm_response


if __name__ == "__main__":
    result = full_pipeline_inference(
        prompt_name = "/Users/sv/Desktop/Desktop - S’s MacBook Pro/discord_search_bot/prompts/search_prompt.txt", 
        chroma_collection = collection_anduril, 
        user_query = 'What is Steven currently building?',
        limit = 30
    )
    print(f'result: {result}')
'''
client.run(discord_bot_token)
if __name__ == "__main__":
    messages = ast.literal_eval(open('discord_messages_all', 'r', encoding='utf-8').read())
    chunk_documents(messages)
    query = 'What is ohsheetklaus currently building'
    responses = run_hybrid_search(collection = collection_anduril, query = query)
    documents = responses.get('documents', '')[:10]
    print(len(documents))
    system_prompt = retrieve_prompt('search_prompt.txt').format(USER_QUERY = query, DOCS = documents)
    token_count = get_token_count(system_prompt)
    print(f'token length: {token_count}')
    llm_response = answer_query(system_prompt = system_prompt)
    print(f'llm_responses: {llm_response}')
'''