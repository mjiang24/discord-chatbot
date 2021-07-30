from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text
import pickle

import discord
from discord.ext import commands
from discord import FFmpegPCMAudio


intents = discord.Intents.default()
intents.members = True
client = commands.Bot(command_prefix = '$', intents = intents)

MODEL = input("enter model: ")

data = pickle.load(open(f"./processed_data/{MODEL}.pickle", "rb"))
enc_model = load_model(f"./models/{MODEL}/enc_model/", compile=False)
dec_model = load_model(f"./models/{MODEL}/dec_model/", compile=False)
dense = pickle.load(open(f"./models/{MODEL}/dense.pickle", "rb"))
vocab = data["vocab"]
inv_vocab = data["inv_vocab"]

def infer(message):
    prepro1 = message
    prepro1 = clean_text(prepro1)
    prepro = [prepro1]

    txt = []
    for x in prepro:
        lst = []
        for y in x.split():
            try:
                lst.append(vocab[y])
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)

    txt = pad_sequences(txt, 13, padding='post')

    stat = enc_model.predict(txt)
    empty_target_seq = np.zeros((1, 1))

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        decoder_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
        decoder_concat_input = dense(decoder_outputs)

        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '

        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index

        stat = [h, c]

    return decoded_translation

on = False

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.command(pass_context= True)
async def test():
    global on
    on = True




@client.command(pass_context= True)
async def join(ctx):
    print('test')
    if(ctx.author.voice):
        channel =  ctx.message.author.voice.channel
        await channel.connect()

    else:
        await ctx.send("not in voice channel")


@client.command(pass_context= True)
async def leave(ctx):
    if (ctx.voice_client):
        await ctx.guild.voice_client.disconnect()
    else:
        await ctx.send("not in voice channel")

@client.command()
async def play(ctx):

    voice = discord.utils.get(client.voice_clients, guild = ctx.guild)
    print(voice.poll_voice_ws)
    source = FFmpegPCMAudio(executable="C:/FFmpeg/ffmpeg.exe", source="monke.mp3")
    voice.play(source)

@client.event
async def on_message(message):
    global on
    if message.author == client.user:
        return

    if on:
        text = message.content
        result = infer(text)

        await message.channel.send(result)

    if message.content.startswith("$test"):
        on = True





client.run('ODU3NDM4ODEyODQ0ODUxMjAw.YNPmHw.TYtLzVnpbrwCeKsOUzQD-k5DQEY')