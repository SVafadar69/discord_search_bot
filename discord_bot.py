import interactions
from interactions import Client, Intents, slash_command, SlashContext, listen, slash_option, OptionType
from src.search_functions import (full_pipeline_inference)
from src.search_functions import (collection_anduril)
from dotenv import load_dotenv
import os 
import asyncio 
import uvicorn 

app = FastAPI()
bot = Client(intents = Intents.ALL)

prompt_name = f'{os.path.join(os.getcwd(), "prompts/search_prompt.txt")}'
print(f'prompt_name: {prompt_name}')

@app.get('/')
async def health():
    return {'status': 'ok'}

@listen
async def on_ready():
    print('Ready')

@slash_command(name="query", description="What do you want to know")
@slash_option(
    name = 'input_text', 
    description = 'input_text', 
    required = True, 
    opt_type = OptionType.STRING
)

async def get_response(ctx: SlashContext, input_text: str): 
    await ctx.defer()
    # replace data_querying with Chroma 
    response = full_pipeline_inference(prompt_name = prompt_name, chroma_collection = collection_anduril, user_query = input_text)
    response = f'**Input Query**: {input_text}\n\n{response}'
    await ctx.send(response)

@slash_command(name='updateddb', description='Update your the chroma db')
async def updated_database(ctx: SlashContext): 
    await ctx.defer()
    update = await update_index() # replace with your own 
    if update: response = f'Updated {sum(update)} document chunks'
    else: response = f'Error updating index'
    await ctx.send(response)

async def main():
    config = uvicorn.Config(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)), reload=False)
    server = uvicorn.Server(config)

    await asyncio.gather(
        server.serve(),
        bot.start(os.getenv('DISCORD_BOT_TOKEN'))
    )


if __name__ == "__main__": 
    asyncio.run(main())
    

