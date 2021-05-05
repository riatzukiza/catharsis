require('dotenv').config()

const Discord = require('discord.js');
const client = new Discord.Client();

import * as fs from 'fs'

const chatLogFile = fs.createWriteStream("./corpus/discord.txt",{ flags:"a" })

client.on('ready', () => {
  console.log(`Logged in as ${client.user.tag}!`);
});

client.on('message', msg => {
    chatLogFile.write(msg.content + "\n")
});


client.login(process.env.DISCORD_KEY);