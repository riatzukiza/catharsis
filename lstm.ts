import * as tf from "@tensorflow/tfjs-node-gpu"
import * as fs from 'fs'

const CHAR_SET_SIZE = 85;
const LSTM_LAYER_SIZE = 32;
const SAMPLE_LENGTH = 512;
const SAMPLE_STEP = SAMPLE_LENGTH;
const NUM_EPOCS = 1;
const NUM_ERA  = 1;
const BATCH_SIZE = 512 ;
const LENGTH = 512;
const TEMPERATURE = 0.05;
const EXAMPLES_PER_EPOC = 10


// Avoid overwriting the original input.

// const encodeDataSet = (
//     chars:string[],
//     charSetSize = CHAR_SET_SIZE
// ) => tf.oneHot(
//     tf.tensor1d(
//         chars.map(c => c.charCodeAt(0)),
//         'int32'
//     ),
//     charSetSize
// );
export function create() {

    const model = tf.sequential();

    const lstm1 = tf.layers.lstm({
        units: LSTM_LAYER_SIZE,
        inputShape:  [SAMPLE_LENGTH, CHAR_SET_SIZE] ,
        returnSequences: true,
    });

    const lstm2 = tf.layers.lstm({
        units: LSTM_LAYER_SIZE,
        returnSequences: false,
    });

    model.add(lstm1);
    model.add(lstm2);

    model.add(tf.layers.dense({
        units: CHAR_SET_SIZE, activation: 'softmax'
    }));

    const optimizer = tf.train.rmsprop(0.05);
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    return model
}

function createOneHotFromSampleData(corpus,vocab = []) {
}

function encode() {
}

export async function train(model:tf.Sequential, {
    epochs = NUM_ERA,
    batchSize = BATCH_SIZE,
    examplesPerEpoch = EXAMPLES_PER_EPOC,
    text
}) {
    console.log("training", "*".repeat(i))

    //   for (let e of sentences.entries()) {
    //     var i = e[0]
    //     var sentence = e[1]
    //     for (let e2 of sentence.split(" ").entries()) {
    //       var t = e2[0]
    //       var word = e2[1]
    //       X.set(i, t, word_indices[word], 1)
    //     }
    //     y.set(i, word_indices[next_words[i]], 1)
    //   }

    // factor this out
    // XS === training data
    // YS === label data

    // x shape =  [numExamples, this.sampleLen_, this.charSetSize_]
    // y shape = [numExamples, this.charSetSize_]

    // const [xs, ys] = data.nextDataEpoch(examplesPerEpoch)
    let trainingDataExample = [
        [
            //"a","b","c","d","e","f","g"
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
        ],
    ]
    let labelDataExample = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]


    const trainingData = tf.oneHot(tf.tensor1d(), Math.pow(2, 10))
    const labelData = tf.oneHot()

    await model.fit(xs, ys, { 
        epochs, 
        batchSize, 
        stepsPerEpoch:examplesPerEpoch
    });

    xs.dispose();
    ys.dispose();
}

function generateVocabulary() {
}

/**
 * 
 * Create a onehot encoded vector from a text given a vocabulary
 * @param text 
 */
function parseText(text, vocab):string[] {
    return []
}

function extractCharacterSet() {
}

/**
 * 
 * Generate a single character (Timestep) from an lstm model
 * @param model LSTM onehot encoded language model
 * @param charset valid characters in the language
 */
function generateCharacter(model:tf.Sequential, charset) {
}

/**
 * 
 * Generate a sampling of text from an lstm model.
 * @param model LSTM onehot encoded language model
 */
function generate (model) {
}

function sample(probs:tf.Tensor, temperature:number) {
    return tf.tidy(() => {
        const logits = <tf.Tensor1D>tf.div(tf.log(probs), Math.max(temperature, 1e-6));
        const isNormalized = false;
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}

(async () => {

    const corpus = fs.readFileSync('./corpus/meditations.mb.txt','utf8')+ "\n" 
        + fs.readFileSync('./corpus/cybercft.txt','utf8');

    const model = create()

    await train(model,{
        text:corpus
    })

    console.log("sample",generate(model,))
})()
