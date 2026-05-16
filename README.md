# NeedleJS

Run the [Cactus Compute Needle](https://github.com/cactus-compute/needle) tool-calling model from JavaScript — no Python, no server, no model hosting. The ONNX weights and SentencePiece tokenizer ship inside this package, so `npm install needlejs` is everything you need.

Needle is a 26M-parameter encoder-decoder transformer that takes a natural-language query and a list of tool definitions and returns a JSON tool call. NeedleJS ports it to [onnxruntime-web](https://github.com/microsoft/onnxruntime) (which runs in both Node and the browser via WebAssembly).

## How it works

```
User query + tools JSON
         │
         ▼
  NeedleTokenizer          (SentencePiece BPE, vocab 8192)
         │
         ▼
  needle_encoder.onnx      (12-layer encoder, d_model=512)
         │
         ▼
  needle_decoder.onnx      (8-layer decoder, autoregressive)
    + ConstrainedDecoder   (trie + JSON state machine keeps output valid)
         │
         ▼
  [{"name":"tool_name","arguments":{...}}]
```

## Quickstart

```js
import { Needle } from 'needlejs';

const needle = new Needle();
await needle.load();  // uses bundled model files

const tools = [{
  name: 'get_weather',
  description: 'Get current weather for a city.',
  parameters: {
    location: { type: 'string', description: 'City name.' },
  },
}];

const result = await needle.generate('What is the weather in San Francisco?', tools);
// '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]'
```

`load()` resolves the bundled model files relative to the installed package and reads them from disk. Pass `encoderPath`, `decoderPath`, `tokenizerPath`, or `vocabPath` to override.

## JavaScript API

### `Needle` (high-level facade)

```js
import { Needle } from 'needlejs';

const needle = new Needle();

// Load using bundled files (default), or override any subset:
await needle.load({
  // encoderPath, decoderPath, tokenizerPath, vocabPath
  // useWebGPU: true,  // try the WebGPU EP first, fall back to wasm
});

const json = await needle.generate(query, tools, {
  maxGenLen: 256,    // max tokens to generate
  maxEncLen: 1024,   // max encoder input length
  constrained: true, // enable JSON-constrained decoding
  onToken: (piece) => process.stdout.write(piece), // streaming callback
});
```

### `NeedleModel` + `NeedleTokenizer` (low-level)

```js
import { NeedleModel, NeedleTokenizer, generate, bundledModelPaths } from 'needlejs';

const { encoderPath, decoderPath, tokenizerPath, vocabPath } = bundledModelPaths();

const model = new NeedleModel();
await model.loadFromPaths(encoderPath, decoderPath);

const tokenizer = await NeedleTokenizer.fromPath(tokenizerPath, vocabPath);

const result = await generate(model, tokenizer, query, tools);
```

## Regenerating the ONNX model files

The model files are pre-built and shipped with the package. To regenerate them (e.g. after a Needle upstream release):

```bash
# Install Python deps
pip install -r scripts/requirements_export.txt
pip install git+https://github.com/cactus-compute/needle

# Export (downloads checkpoint from HuggingFace, ~52 MB fp16 output)
python scripts/export_onnx.py --fp16 --validate --output-dir models/
```

This writes to:

```
models/
├── needle_encoder_fp16.onnx   (~28 MB)
├── needle_decoder_fp16.onnx   (~43 MB)
└── tokenizer/
    ├── needle.model
    └── needle.vocab
```

`--validate` runs onnxruntime against the PyTorch reference and asserts the max absolute difference is under 0.05 (fp16 tolerance). `--fp16` halves model size with negligible accuracy loss.

### Export options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `../models` | Where to write the ONNX files |
| `--fp16` | off | Convert to float16 after export |
| `--validate` | off | Validate output against PyTorch |
| `--checkpoint` | (HuggingFace) | Path to a local `checkpoint.pkl` |

## Development

```bash
npm install
npm test       # runs unit tests + end-to-end against the bundled models
npm run build  # build ESM + UMD library bundles to dist/
```

### Project layout

```
needlejs/
├── scripts/
│   ├── export_onnx.py          # Python: JAX/Flax → PyTorch → ONNX
│   └── requirements_export.txt
├── src/
│   ├── tokenizer.js            # SentencePiece wrapper + tool name normalization
│   ├── constrained.js          # JSON-constrained decoding (Trie + state machine)
│   ├── model.js                # onnxruntime-web encoder/decoder sessions
│   ├── generator.js            # Autoregressive generation loop
│   └── index.js                # Public API + Needle facade
├── test/
│   ├── constrained.test.ts     # Trie, JsonStateMachine, logit masking
│   ├── tokenizer.test.ts       # snake_case conversion, tool normalization
│   └── needle.test.ts          # End-to-end against bundled models
└── models/                     # ONNX + tokenizer files shipped with the package
```

## Technical notes

**Why PyTorch re-implementation instead of jax2tf?**
Flax's `nn.scan` compiles to `while_loop` in XLA, which becomes an ONNX loop node. ONNX loop nodes perform poorly in onnxruntime-web's WASM backend. Re-implementing the model in PyTorch produces unrolled layers and a flat, fast ONNX graph.

**Grouped-query attention**
Needle uses 8 query heads and 4 KV heads. The JS model wrapper repeats K/V tensors (`repeat_interleave`) to match the query head count, matching the Python implementation exactly.

**Constrained decoding**
The decoder is constrained to only produce valid JSON matching the provided tool schema. A character-level trie over tool names and parameter keys, combined with a JSON state machine that tracks buffer context, masks invalid tokens to `-Infinity` before each argmax. This is a direct port of `needle/model/constrained.py`.

**Vocab table source**
The WASM build of `sentencepiece-js` doesn't expose `GetPieceSize` / `IdToPiece`, so constrained decoding reads the companion `needle.vocab` file (shipped alongside `needle.model`) to build its token-string table.

## License

MIT
