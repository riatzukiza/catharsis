import * as tf from "@tensorflow/tfjs-node-gpu"
import * as fs from 'fs'
import * as tf from "@tensorflow/tfjs-node-gpu"
import * as fs from 'fs'

// class Model {
//     constructor({
//         path = "./models/default",
//         data = "",
//         parameters,
//     }) {}
//     public train() {}
//     public generate() {}
//     public save() {}
// }

const LSTM_LAYER_SIZE = 32;
const SAMPLE_LENGTH = 128;
const SAMPLE_STEP = SAMPLE_LENGTH;
const NUM_EPOCS = 1;
const NUM_ERA  = 5;
const BATCH_SIZE = 512 ;
const LENGTH = 128;
const TEMPERATURE = 0.05;
const EXAMPLES_PER_EPOC = 10;

import CharacterSet from "./charset"

const Models = {}

class LSTM {
    model:tf.LayersModel

    static create({
        sampleLength = SAMPLE_LENGTH,
        charsetSize = 128
    }) {
        // define the models layer structure
        const model = tf.sequential();
        const lstm1 = tf.layers.lstm({
            units: LSTM_LAYER_SIZE,
            inputShape: [sampleLength, charsetSize],
            returnSequences: true,
        });

        // const lstm2 = tf.layers.lstm({
        //     units: LSTM_LAYER_SIZE,
        //     returnSequences: false,
        // });

        const optimizer = tf.train.rmsprop(0.05);


        model.add(lstm1);
        //model.add(lstm2);
        model.add(tf.layers.dense({ units: charsetSize, activation: 'softmax' }));

        model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

    }
    constructor({
        model = tf.sequential()
    }) {}

}

export function create() {
    // define the models layer structure
    const model = tf.sequential();
    const lstm1 = tf.layers.lstm({
        units: LSTM_LAYER_SIZE,
        inputShape: [SAMPLE_LENGTH, charset.length],
        returnSequences: true,
    });

    const lstm2 = tf.layers.lstm({
        units: LSTM_LAYER_SIZE,
        returnSequences: false,
    });

    const optimizer = tf.train.rmsprop(0.05);


    model.add(lstm1);
    //model.add(lstm2);
    model.add(tf.layers.dense({ units: charset.length, activation: 'softmax' }));

    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
}

export function encode(text, {
    sampleLength = SAMPLE_LENGTH,
    charset = CharacterSet.create(text)
}) {


    const trainingData =  tf.buffer([text.length,sampleLength,charset.size])
    const labelData = tf.buffer([text.length,charset.size])

    console.log("generating training data")

    for(let i = 0; i < text.length ; i++) {
        for(let j = 0; j < SAMPLE_LENGTH; j++) {
            let k = charset.getIndexFromChar(text[i+j])
            trainingData.set(1,i,j,k)
        }
        labelData.set(1,i,charset.getIndexFromChar(text[i]))
    }

    return [trainingData,labelData]

}
export function train() {}
export function generate() {}
export function save() {}
export function load(path) {}

