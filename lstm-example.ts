import * as tf from "@tensorflow/tfjs-node-gpu"
import * as Path from "path"
import * as fs from 'fs'
import ProgressBar from 'progress'

const START_AT = 14
const CHAR_SET_SIZE = 128;
const SAMPLE_LENGTH = 32;
const NUM_EPOCS = 10;
const BATCH_SIZE = 128 ;
const LENGTH = 128;
const TEMPERATURE = 0.75;
const CHUNK_SIZE = Math.pow(2,12)
// const EXAMPLES_PER_EPOC = 10;
// const LSTM_LAYER_COUNT = 5;
// const ACTIVATION_LAYERS = 2;
// const MODEL_PATH = `file://./${SAMPLE_LENGTH}_${BATCH_SIZE}_${LSTM_LAYER_COUNT}_${ACTIVATION_LAYERS}`;
const MODEL_PATH = `file://./lstm_${SAMPLE_LENGTH}_${BATCH_SIZE}_${CHAR_SET_SIZE}`;

async function train(chars,charmap,charset) {
    const corpus = chars
    const trainingData =  tf.buffer([chars.length,SAMPLE_LENGTH,CHAR_SET_SIZE])
    const labelData = tf.buffer([chars.length,CHAR_SET_SIZE])

    console.log("digesting training data")

    for(let i = 0; i < chars.length ; i++) {
        for(let j = 0; j < SAMPLE_LENGTH; j++) {
            let k = charmap[chars[i+j]]
            trainingData.set(1,i,j,k)
        }
        labelData.set(1,i,charmap[chars[i]])
    }
    console.log("training data full digested (YUM)")


    const optimizer = tf.train.rmsprop(0.05);
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
            returnSequences: true,
        });

        const lstm2 = tf.layers.lstm({
            units: CHAR_SET_SIZE,
            returnSequences: false,
        });

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
        model.add(lstm2);
        // for(let layer of innerLSTMLayers) {
        //     model.add(layer);
        // }

        model.add(tf.layers.dense({ units: CHAR_SET_SIZE, activation: 'softmax' }));
        // for(let i = 0; i < ACTIVATION_LAYERS; i++) {
        //     model.add(tf.layers.dense({ units: CHAR_SET_SIZE, activation: 'softmax' }));
        // }

        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
        console.log("")

        console.log("Saving Model")
        await model.save(MODEL_PATH)
        console.log("Model saved")
    }

    const xs = trainingData.toTensor()
    const ys = labelData.toTensor()

    console.log("initiate training sequence")

    await model.fit(xs, ys, {
        epochs: NUM_EPOCS,
        batchSize: BATCH_SIZE,
        validationStep:0.25
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


    console.log("Generating sample from model at a temperature of",TEMPERATURE)
    // take a random slice out of the training data as a seed
    let startIndex = Math.round(Math.random() * (chars.length - SAMPLE_LENGTH - 1))
    let textSlice = corpus.slice(startIndex, startIndex+ SAMPLE_LENGTH)
    let seedSentance = textSlice
    let generated = ""
    let sentenceIndices = textSlice.split("").map((c) => charmap[c])

    let sampleGenerationProgress = new ProgressBar("generating",{
        total:LENGTH
    })

    while (generated.length < LENGTH) {
        // Encode the current input sequence as a one-hot Tensor.
        const inputBuffer = tf.buffer([1, SAMPLE_LENGTH, CHAR_SET_SIZE]);

        for (let i = 0; i < SAMPLE_LENGTH; ++i) {
            inputBuffer.set(i, 0, 1, sentenceIndices[i]);
        }

        const input = inputBuffer.toTensor();


        // Call model.predict() to get the probability values of the next
        // character.
        const output = <tf.Tensor<tf.Rank>>model.predict(input);

        // Sample randomly based on the probability values.
        const winnerIndex = sample(tf.squeeze(output), TEMPERATURE);
        const winnerChar = charset[winnerIndex];

        generated += winnerChar;
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);


        // Memory cleanups.
        input.dispose();
        output.dispose();
        sampleGenerationProgress.tick()
    }
    console.log("generated text:",generated)
}

async function chunkedTraining(chars,charmap,charset) {
    const chunkCount = Math.ceil(chars.length/CHUNK_SIZE)
    console.log("total chunks",chunkCount)
    for(let i = START_AT; i < chunkCount; i++) {
        console.log({CHUNK_SIZE})
        console.log("total characters",chars.length)
        console.log("training on chunk",i,"of",chunkCount)
        let chunk;
        if(chunkCount*(i+1) < chars.length) {
            chunk = chars.slice(CHUNK_SIZE*i,CHUNK_SIZE*(i+1))
        } else {
            chunk = chars.slice(CHUNK_SIZE*i,chars.length)
        }

        console.log("characters in chunk",chunk.length)
        console.log("data:",chunk)
        await train(chunk,charmap,charset)
    }
}

(async () => {

    // read the corpus in from a few files
    console.log("loading corpus")

    const corpus = (
        fs.readFileSync('./corpus/meditations.mb.txt','utf8') + "\n" +
            fs.readFileSync('./texts/twitter/minecraft.twitter.txt','utf8') +"\n"+
            fs.readFileSync('./texts/twitter/chicken.twitter.txt','utf8')
    ) .replace(/(?:https?|ftp):\/\/[\n\S]+/g, '');

    console.log("loaded corpus")

    // be smarter about this
    //const chars = corpus.split("") // split corpus into an array of characters
    const chars = corpus
    const charmap = {} // a map of characters -> indexes

    console.log("creating charset")
    let charset = []
    for(let c of corpus) {
        if(charmap[c]) continue;
        charmap[c] = charset.length
        charset.push(c)
    }
    console.log("//charset created//////////////")

    await chunkedTraining(chars,charmap,charset)


})()
