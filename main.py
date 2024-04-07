import asyncio
import math
import os
import discord
import dotenv
import mai
from datetime import datetime

dotenv.load_dotenv()

CHAT_HISTORY_SINCE = datetime(2024, 3, 25)

class MyClient(discord.Client):
    mai = None
    chat_since = None

    async def on_ready(self):
        print('Logged on as', self.user)
        if MyClient.chat_since is None:
            MyClient.chat_since = datetime.now()
        if MyClient.mai is None:
            MyClient.mai = mai.Flan()

    async def on_message(self, message):
        print("Perceived message from %s. "%message.author)
        

        # don't respond to ourselves
        if message.author == self.user:
            print("Ignoring self...")

        elif self.user in message.mentions:
            print("Directed to %s. Answering..."%self.user)
            answer = MyClient.mai.tell(message.clean_content)
            nchunks = math.ceil(len(answer) / 2000)
            
            for i in range(nchunks):
                is_first_message = i == 0
                chunk = answer[i*2000:(1+i)*2000]
                if is_first_message:
                    await message.reply(chunk, mention_author=True)
                else:
                    await message.channel.send(chunk)



intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(os.getenv("DISCORD_TOKEN"))
