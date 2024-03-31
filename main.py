import asyncio
import os
import discord
import dotenv
import codellama
from datetime import datetime

dotenv.load_dotenv()

CHAT_HISTORY_SINCE = datetime(2024, 3, 25)
SYS_PROMPT = """You are a friendly and helpful assistant programmer."""

class MyClient(discord.Client):
    chat_since = None

    async def on_ready(self):
        print('Logged on as', self.user)
        if MyClient.chat_since is None:
            MyClient.chat_since = datetime.now()

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        if self.user in message.mentions:
            ai_reply = codellama.run(message, [], SYS_PROMPT)
            await message.reply(ai_reply, mention_author=True)

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(os.getenv("DISCORD_TOKEN"))