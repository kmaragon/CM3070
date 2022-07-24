const express = require('express');
const tf = require('@tensorflow/tfjs-node-gpu');
const mvae = require('@magenta/music/node/music_vae');
const core = require('@magenta/music/node/core');
const easymidi = require('easymidi');
const prompt = require('prompt');
const fs = require('fs');

class GeneratorBase {
    constructor(model, temperature) {
        this.model = model;
        this.temperature = temperature
    }

    initialize() {
        throw new Error("Not Implemented");
    }

    async sample(sampleSize) {
        await this.initialize();
        const samples = await this.model.sample(sampleSize, this.temperature);

        return samples.map(sample => {
            const samplesPerQuarter = sample.quantizationInfo.stepsPerQuarter;
            const quartersPerMinute = sample.tempos[0].qpm;

            const samplesPerMinute = (quartersPerMinute * samplesPerQuarter);
            const sampleLengthSeconds = 60.0 / samplesPerMinute;

            const sampleLengthMilliseconds = sampleLengthSeconds * 1000;
            return sample.notes.map(note => {
                return {
                    pitch: note.pitch,
                    start: sampleLengthMilliseconds * note.quantizedStartStep,
                    end: sampleLengthMilliseconds * note.quantizedEndStep
                }
            })
        });
    }

    play(sampleCollection) {

        const startOffset = (new Date()).getTime() + 500;
        const promises = [];

        let s = 0;
        for (let samples of sampleCollection) {
            const track = ++s;

            const output = new easymidi.Output(`Track${track}`, true);

            const trackPromise = new Promise((success, _) => {
                for (let i = 0; i < samples.length; i++) {
                    const note = samples[i];

                    const startTime = startOffset + note.start;
                    const now = new Date().getTime();
                    const isLast = i === (samples.length - 1);
                    const isFirst = i === 0;

                    // set the start note
                    setTimeout(() => {
                        if (isFirst) {
                            console.log(`Starting Track ${track}`);
                        }

                        output.send("noteon", {
                            note: note.pitch,
                            velocity: 127,
                            channel: track
                        });

                        // set the end note
                        setTimeout(() => {
                            output.send("noteoff", {
                                note: note.pitch,
                                velocity: 127,
                                channel: track
                            });

                            if (isLast) {
                                output.close();
                                success();
                            }
                        }, note.end - note.start);
                    }, startTime - now);
                }
            });

            promises.push(trackPromise);
        }

        return Promise.all(promises).then(() => {});
    }
}

class VAEGenerator extends GeneratorBase {
    static PORT = 3005;

    static _staticInit = function () {
        // make sure fetch works when we load MusicVAE
        const globalAny = global;
        globalAny.performance = Date;
        globalAny.fetch = require('node-fetch');

        // set up a global express app to serve the model file
        const express_app = express();
        express_app.use(express.static(process.cwd()));

        return new Promise((success, failure) => {
            express_app.listen(VAEGenerator.PORT, success);
        });
    }();

    async _onConstruct() {
        await VAEGenerator._staticInit;
        await this.model.initialize();
    }

    constructor(temperature = 0.5) {
        super(new mvae.MusicVAE(`http://localhost:${VAEGenerator.PORT}/checkpoint`), temperature);
        this._instanceInit = this._onConstruct();
    }

    async initialize() {
        return this._instanceInit;
    }

}

const gen = new VAEGenerator(0.35);

let playSample;
let batchGenerateOutput;
let batchGenerateCount;

for (let i = 2; i < process.argv.length; i++)
{
    const arg = process.argv[i];
    if (arg === '--play') {
        const file = process.argv[++i];
        if (!file) {
            console.error("missing argument for --play");
            process.exit(-1);
        }

        const score = JSON.parse(fs.readFileSync(file));
        if (score) {
            playSample = score;
        }

        continue;
    }

    if (arg === '--count') {
        batchGenerateCount = parseInt(process.argv[++i]);
    }

    if (arg === '--batch') {
        const batchOutputDir = process.argv[++i];
        if (!batchOutputDir || !fs.existsSync(batchOutputDir)) {
            console.error('Batch output directory not valid');
            process.exit(-1);
        }

        if (!fs.statSync(batchOutputDir).isDirectory()) {
            console.error('Batch output directory not a directory');
            process.exit(-1);
        }

        batchGenerateOutput = batchOutputDir;
    }
}

if (playSample) {
    gen.play(playSample).then(() => {
        process.exit(0);
    });
} else if (batchGenerateOutput) {
    if (!batchGenerateCount) {
        console.error('Batch Generate requires --count with non-zero count');
        process.exit(-1);
    }

    let filenameIndex = 0;

    let lastPromise = new Promise((success, _) => { success(); });
    for (let i = 0; i < batchGenerateCount; i++) {
        // first figure out a number of samples
        const samples = Math.round(3 + (Math.random() * 2.5));

        lastPromise = lastPromise.then(() => {
            // find a filename
            let filename;
            while (true) {
                filename = `${batchGenerateOutput}/${filenameIndex.toString().padStart(5, '0')}.json`;
                if (!fs.existsSync(filename)) {
                    break;
                }

                ++filenameIndex;
            }

            return gen.sample(samples).then(music => {
                console.log(`Writing file ${filename}`);
                fs.writeFileSync(filename, JSON.stringify(music));
            });
        });
    }

    lastPromise.then(() => {
        console.log('Done!');
        process.exit(0);
    })
} else {
    gen.sample(4).then(music => {
        gen.play(music).then(() => {
            const pmessage = 'Save to file (leave blank to discard): ';
            prompt.get([pmessage], (err, result) => {
                if (err) {
                    console.error(err);
                    process.exit(-1);
                }

                if (!result[pmessage]) {
                    process.exit(0)
                }

                console.log(`Writing file ${result[pmessage]}`);
                fs.writeFileSync(result[pmessage], JSON.stringify(music));
                process.exit(0)
            });
        });
    });
}