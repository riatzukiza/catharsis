import {TextData} from './text-data'
import * as fs from 'fs'

let textData = new TextData(
    "data",
    (fs.readFileSync('./corpus/meditations.mb.txt','utf8') + "\n" +
        fs.readFileSync('./texts/twitter/minecraft.twitter.txt','utf8') +"\n"+
        fs.readFileSync('./texts/twitter/chicken.twitter.txt','utf8'))
        .replace(/(?:https?|ftp):\/\/[\n\S]+/g, ''),
    64,64
)

console.log(textData.nextDataEpoch(5))
