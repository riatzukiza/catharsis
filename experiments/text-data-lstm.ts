import * as tf from "@tensorflow/tfjs-node-gpu"
import * as Path from "path"
import * as fs from 'fs'

import ProgressBar from 'progress'
import {TextData} from '../text-data'

const START_AT = 0
const CHAR_SET_SIZE = 256;
const SAMPLE_LENGTH = 32;
const NUM_EPOCS = 20;
const BATCH_SIZE = 64;
const LENGTH = 128;
const TEMPERATURE = 0.75;
const CHUNK_SIZE = Math.pow(2,12)
const EXAMPLES_PER_EPOC = 10000;
// const LSTM_LAYER_COUNT = 5;
// const ACTIVATION_LAYERS = 2;
// const MODEL_PATH = `file://./${SAMPLE_LENGTH}_${BATCH_SIZE}_${LSTM_LAYER_COUNT}_${ACTIVATION_LAYERS}`;
const MODEL_PATH = `file://./models/lstm_${SAMPLE_LENGTH}_${BATCH_SIZE}_${CHAR_SET_SIZE}`;

const corpus = (
    fs.readFileSync('./corpus/meditations.mb.txt','utf8') + "\n" +
        fs.readFileSync('./texts/twitter/minecraft.twitter.txt','utf8') +"\n"+
        fs.readFileSync('./texts/twitter/chicken.twitter.txt','utf8')
) .replace(/(?:https?|ftp):\/\/[\n\S]+/g, '');

// be smarter about this
//const chars = corpus.split("") // split corpus into an array of characters
const chars = []
const charmap = {} // a map of characters -> indexes
const charCounts = {}

console.log("creating charset")
let charset = []
for(let c of corpus) {
    if(charCounts[c]) {charCounts[c]++}
    else {charCounts[c] = 1}
    if(charset.length > CHAR_SET_SIZE) break;
    if(charmap[c]) continue;
    charmap[c] = charset.length
    charset.push(c)
    // console.log({charmap,charset})
}
for(let c of corpus) {
    if(!charmap[c]) continue;
    chars.push(c)
}
//console.log("//charset created//////////////",{chars,charmap,charset,charCounts,})

const textData = new TextData("data", chars.join(''), SAMPLE_LENGTH,64)
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
}

async function train(chars,charmap,charset) {
    const optimizer = tf.train.rmsprop(0.001);
    let model ;
    try {
        console.log("loading model")
        model = await tf.loadLayersModel(MODEL_PATH+"/model.json")
        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
        console.log("Model successfully loaded")
    } catch(err) {
        console.log("failed to load model, building a new one",err)
        // define the models layer structure
        model = tf.sequential();

        const lstm1 = tf.layers.lstm({
            units: CHAR_SET_SIZE,
            inputShape: [SAMPLE_LENGTH, CHAR_SET_SIZE],
            //returnSequences: true,
            returnSequences: false,
        });

        // const lstm2 = tf.layers.lstm({
        //     units: CHAR_SET_SIZE,
        //     returnSequences: false,
        // });

        const innerLSTMLayers = []

        // for(let i = 0; i < LSTM_LAYER_COUNT-1; i++) {
        //     innerLSTMLayers.push(
        //         tf.layers.lstm({
        //             units: CHAR_SET_SIZE,
        //             inputShape: [SAMPLE_LENGTH, CHAR_SET_SIZE],
        //             returnSequences: (i < LSTM_LAYER_COUNT-1),
        //         })) ;
        // }

        model.add(lstm1);
        //model.add(lstm2);
        // for(let layer of innerLSTMLayers) {
        //     model.add(layer);
        // }

        model.add(tf.layers.dense({ units: CHAR_SET_SIZE, activation: 'softmax' }));
        // for(let i = 0; i < ACTIVATION_LAYERS; i++) {
        //     model.add(tf.layers.dense({ units: CHAR_SET_SIZE, activation: 'softmax' }));
        // }

        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

        console.log("Saving Model")
        await model.save(MODEL_PATH)
        console.log("Model saved")
    }

    const [xs, ys] = textData.nextDataEpoch(EXAMPLES_PER_EPOC)

    console.log("initiate training sequence")
    await model.fit(xs, ys, {
        epochs: NUM_EPOCS,
        batchSize: BATCH_SIZE,
        //validationStep:0.25
        //stepsPerEpoch: EXAMPLES_PER_EPOC
    });

    xs.dispose();
    ys.dispose();

    console.log("Saving model")
    await model.save(MODEL_PATH)
    function sample(probs: tf.Tensor, temperature: number) {
        return tf.tidy(() => {
            const logits = <tf.Tensor1D>tf.div(tf.log(probs), Math.max(temperature, 1e-6));
            const isNormalized = false;
            // `logits` is for a multinomial distribution, scaled by the temperature.
            // We randomly draw a sample from the distribution.
            return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
        });
    }


    // take a random slice out of the training data as a seed
    let startIndex = Math.round(Math.random() * (chars.length - SAMPLE_LENGTH - 1))
    let textSlice = corpus.slice(startIndex, startIndex+ SAMPLE_LENGTH)
    let seedSentance = ""
    let generated = ""
    let sentenceIndices = textSlice.split("").map((c) => charmap[c])
    // console.log("Generating sample from model ",{
    //     TEMPERATURE,
    //     startIndex,
    //     seedSentance,
    //     sentenceIndices
    // })

    let sampleGenerationProgress = new ProgressBar(":bar",{
        total:LENGTH
    })

    while (generated.length < LENGTH) {
        // Encode the current input sequence as a one-hot Tensor.
        const inputBuffer = tf.buffer([1, SAMPLE_LENGTH, CHAR_SET_SIZE]);

        for (let i = 0; i < SAMPLE_LENGTH; ++i) {
            inputBuffer.set(1, 0, i, sentenceIndices[i]);
        }

        const input = inputBuffer.toTensor();
        //input.print()


        // Call model.predict() to get the probability values of the next
        // character.
        const output = <tf.Tensor<tf.Rank>>model.predict(input);
        //output.print()

        // Sample randomly based on the probability values.
        const winnerIndex = sample(tf.squeeze(output), TEMPERATURE);
        const winnerChar = charset[winnerIndex];
        //console.log({input,output,winnerIndex,winnerChar})

        generated += winnerChar;
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);


        // Memory cleanups.
        input.dispose();
        output.dispose();
        sampleGenerationProgress.tick()
    }
    console.log({generated})
}

async function chunkedTraining(chars,charmap,charset) {
    const chunkCount = Math.ceil(chars.length/CHUNK_SIZE)
    //console.log("total chunks",chunkCount)
    for(let i = START_AT; i < chunkCount; i++) {
        //console.log({CHUNK_SIZE})
        //console.log("total characters",chars.length)
        //console.log("training on chunk",i,"of",chunkCount)
        const randomIndex = getRandomInt(0,chunkCount)
        let chunk;
        if(chunkCount*(randomIndex+1) < chars.length) {
            chunk = chars.slice(CHUNK_SIZE*randomIndex,CHUNK_SIZE*(randomIndex+1))
        } else {
            chunk = chars.slice(CHUNK_SIZE*randomIndex,chars.length)
        }

        //console.log("characters in chunk",chunk.length)
        //console.log("data:",chunk)
        await train(chunk,charmap,charset)
    }
}

(async () => {

    // read the corpus in from a few files
    console.log("loading corpus")


    console.log("loaded corpus")


    await chunkedTraining(chars.join(""),charmap,charset)


})()
