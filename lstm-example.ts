import * as tf from "@tensorflow/tfjs-node-gpu"
import * as Path from "path"
import * as fs from 'fs'

//const CHAR_SET_SIZE = 85;
const SAMPLE_LENGTH = 32;
const NUM_EPOCS = 100;
const BATCH_SIZE = 128 ;
const LENGTH = 128;
const TEMPERATURE = 0.05;
// const EXAMPLES_PER_EPOC = 10;
// const LSTM_LAYER_COUNT = 5;
// const ACTIVATION_LAYERS = 2;
// const MODEL_PATH = `file://./${SAMPLE_LENGTH}_${BATCH_SIZE}_${LSTM_LAYER_COUNT}_${ACTIVATION_LAYERS}`;
const MODEL_PATH = `file://./${SAMPLE_LENGTH}_${BATCH_SIZE}`;

(async () => {

    // read the corpus in from a few files
    console.log("loading corpus")
    const corpus = fs.readFileSync('./corpus/meditations.mb.txt','utf8')
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

    const trainingData =  tf.buffer([chars.length,SAMPLE_LENGTH,charset.length])
    const labelData = tf.buffer([chars.length,charset.length])

    console.log("generating training data")

    for(let i = 0; i < chars.length ; i++) {
        for(let j = 0; j < SAMPLE_LENGTH; j++) {
            let k = charmap[chars[i+j]]
            trainingData.set(1,i,j,k)
        }
        labelData.set(1,i,charmap[chars[i]])
    }
    console.log("training data generated")


    const optimizer = tf.train.rmsprop(0.05);
    let model ;
    try {
        console.log("loading model")
        model = await tf.loadLayersModel(MODEL_PATH+"/model.json")
        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    } catch(err) {
        console.log("failed to load model, generating new one",err)
        // define the models layer structure
        model = tf.sequential();

        const lstm1 = tf.layers.lstm({
            units: charset.length,
            inputShape: [SAMPLE_LENGTH, charset.length],
            returnSequences: true,
        });

        const lstm2 = tf.layers.lstm({
            units: charset.length,
            inputShape: [SAMPLE_LENGTH, charset.length],
            returnSequences: false,
        });

        const innerLSTMLayers = []
        // for(let i = 0; i < LSTM_LAYER_COUNT-1; i++) {
        //     innerLSTMLayers.push(
        //         tf.layers.lstm({
        //             units: charset.length,
        //             inputShape: [SAMPLE_LENGTH, charset.length],
        //             returnSequences: (i < LSTM_LAYER_COUNT-1),
        //         })) ;
        // }



        model.add(lstm1);
        model.add(lstm2);
        // for(let layer of innerLSTMLayers) {
        //     model.add(layer);
        // }

        model.add(tf.layers.dense({ units: charset.length, activation: 'softmax' }));
        // for(let i = 0; i < ACTIVATION_LAYERS; i++) {
        //     model.add(tf.layers.dense({ units: charset.length, activation: 'softmax' }));
        // }

        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

        console.log("Saving Model")
        await model.save(MODEL_PATH)
    }

    const xs = trainingData.toTensor()
    const ys = labelData.toTensor()

    await model.fit(xs, ys, {
        epochs: NUM_EPOCS,
        batchSize: BATCH_SIZE,
        //stepsPerEpoch: EXAMPLES_PER_EPOC
    });

    xs.dispose();
    ys.dispose();

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
    let seedSentance = textSlice
    let generated = ""
    let sentenceIndices = textSlice.split("").map((c) => charmap[c])

    while (generated.length < LENGTH) {
        // Encode the current input sequence as a one-hot Tensor.
        const inputBuffer = tf.buffer([1, SAMPLE_LENGTH, charset.length]);

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
    }
    console.log("generated text:",generated)
})()
