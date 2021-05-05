require('dotenv').config()
import Twitter from 'twitter'
import * as fs from 'fs'



var client = new Twitter({
    consumer_key: process.env.TWITTER_CONSUMER_KEY,
    consumer_secret: process.env.TWITTER_CONSUMER_SECRET,
    access_token_key: process.env.TWITTER_ACCESS_TOKEN_KEY,
    access_token_secret: process.env.TWITTER_ACCESS_TOKEN_SECRET
});

function scrape(filter:string) {
    const file = fs.createWriteStream(`./texts/twitter/${filter}.twitter.txt`,{
        flags:"a"
    })

    var stream = client.stream('statuses/filter', {track: filter});
    stream.on('data', (event) => {
        if(event.lang === 'en' ) {
            let text = event?.extended_tweet?.full_text || event.text
            if(text.slice(0,2) == "RT") {
                text = text.slice(2)
            }
            console.log(filter,":tweet:",text)
            file.write(text + "\n")
        }
    });

    stream.on('error', (error) => console.error(error));
    return {
        filter,
        file,
        stream
    }
}
function scrapeMany(filters:string[]) {
    const scrapers = []
    for(let filter of filters) {
        scrapers.push(scrape(filter))
    }
    return scrapers
}

scrapeMany([
    "minecraft",
    "chicken"
])
