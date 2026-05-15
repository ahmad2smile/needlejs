# NeedleJS

Run the [Cactus Compute Needle](https://github.com/cactus-compute/needle) tool-calling model in a Chrome extension — no Python, no server, no dependencies at runtime.

Needle is a 26M-parameter encoder-decoder transformer that takes a natural language query and a list of tool definitions and returns a JSON tool call. NeedleJS ports it to [onnxruntime-web](https://github.com/microsoft/onnxruntime) so it runs entirely in WebAssembly inside the browser.

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
  {"name":"tool_name","arguments":{...}}
```

## Quickstart

```js
import { Needle } from 'needlejs';

const needle = new Needle();
await needle.load({
  encoderUrl: 'https://your-cdn.example.com/needle_encoder_fp16.onnx',
  decoderUrl: 'https://your-cdn.example.com/needle_decoder_fp16.onnx',
  tokenizerUrl: 'https://your-cdn.example.com/tokenizer/needle.model',
  wasmDir: 'https://your-cdn.example.com/ort-wasm/',
});

const tools = [{
  name: 'get_weather',
  description: 'Get current weather for a city.',
  parameters: {
    location: { type: 'string', description: 'City name.' },
  },
}];

const result = await needle.generate('What is the weather in San Francisco?', tools);
// '{"name":"get_weather","arguments":{"location":"San Francisco"}}'
```

## Generating the ONNX model files

The ONNX files are not included in this repo. Run the export script once on any machine that has Python and a GPU (or CPU):

```bash
# Install Python dependencies
pip install -r scripts/requirements_export.txt
pip install git+https://github.com/cactus-compute/needle

# Export (downloads checkpoint from HuggingFace, ~52 MB fp16 output)
python scripts/export_onnx.py --fp16 --validate --output-dir models/
```

This produces:

```
models/
├── needle_encoder_fp16.onnx   (~15 MB)
├── needle_decoder_fp16.onnx   (~37 MB)
└── tokenizer/
    └── needle.model
```

`--validate` runs onnxruntime against the PyTorch reference and asserts the max absolute difference is under 0.05 (fp16 tolerance). `--fp16` halves model size with negligible accuracy loss.

### Export options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `../models` | Where to write the ONNX files |
| `--fp16` | off | Convert to float16 after export |
| `--validate` | off | Validate output against PyTorch |
| `--checkpoint` | (HuggingFace) | Path to a local `checkpoint.pkl` |

## JavaScript API

### `Needle` (high-level facade)

```js
import { Needle } from 'needlejs';

const needle = new Needle();

// Load all components at once
await needle.load({ encoderUrl, decoderUrl, tokenizerUrl, wasmDir, useWebGPU });

// Generate a tool call
const json = await needle.generate(query, tools, {
  maxGenLen: 256,    // max tokens to generate
  maxEncLen: 1024,   // max encoder input length
  constrained: true, // enable JSON-constrained decoding
  onToken: (piece) => process.stdout.write(piece), // streaming callback
});
```

### `NeedleModel` + `NeedleTokenizer` (low-level)

```js
import { NeedleModel, NeedleTokenizer, generate, configureOrtWasm } from 'needlejs';

configureOrtWasm('/ort-wasm/');

const model = new NeedleModel();
await model.loadFromUrls(encoderUrl, decoderUrl);

const tokenizer = await NeedleTokenizer.fromUrl(tokenizerUrl);

const result = await generate(model, tokenizer, query, tools);
```

## Chrome Extension

The `extension/` directory is a ready-to-build Chrome Manifest V3 extension.

### Build

```bash
npm install
npm run build:extension   # outputs to dist/extension/
```

Load the `dist/extension/` folder in Chrome via `chrome://extensions` → *Load unpacked*.

### How it works

The extension's service worker (`extension/background.js`) lazily loads the ONNX models from the [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache) on first use. Models are fetched from the HuggingFace Hub if not bundled locally and cached for subsequent uses. Content scripts and the popup communicate with the service worker via `chrome.runtime.sendMessage`:

```js
// From any content script or popup:
chrome.runtime.sendMessage({
  type: 'NEEDLE_GENERATE',
  query: 'Book a flight to Tokyo',
  tools: JSON.stringify(myTools),
}, (response) => {
  if (response.success) console.log(response.result);
});
```

### Bundling models locally

Copy the ONNX files into `models/` before building. The extension build copies them into `dist/extension/models/`. This avoids the first-run download at the cost of a larger extension package (~52 MB fp16).

## Development

```bash
npm install          # install JS dependencies
npm test             # run unit tests (25 tests)
npm run build        # build ESM + UMD library
npm run build:extension  # build Chrome extension bundle
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
├── extension/
│   ├── manifest.json           # Chrome Manifest V3
│   ├── background.js           # Service worker
│   └── popup/                  # Extension popup UI
├── test/
│   ├── constrained.test.js     # Trie, JsonStateMachine, logit masking
│   └── tokenizer.test.js       # snake_case conversion, tool normalization
└── models/                     # ONNX files go here (gitignored)
```

## Technical notes

**Why PyTorch re-implementation instead of jax2tf?**
Flax's `nn.scan` compiles to `while_loop` in XLA, which becomes an ONNX loop node. ONNX loop nodes perform poorly in onnxruntime-web's WASM backend. Re-implementing the model in PyTorch produces unrolled layers and a flat, fast ONNX graph.

**Grouped-query attention**
Needle uses 8 query heads and 4 KV heads. The JS model wrapper repeats K/V tensors (`repeat_interleave`) to match the query head count, matching the Python implementation exactly.

**Constrained decoding**
The decoder is constrained to only produce valid JSON matching the provided tool schema. A character-level trie over tool names and parameter keys, combined with a JSON state machine that tracks buffer context, masks invalid tokens to `-Infinity` before each argmax. This is a direct port of `needle/model/constrained.py`.

**Chrome extension + WASM**
WASM execution in Manifest V3 service workers requires `'wasm-unsafe-eval'` in the extension's content security policy. ORT WASM files must be listed in `web_accessible_resources` and their path communicated to onnxruntime-web via `ort.env.wasm.wasmPaths` before any session is created.

**Service worker lifetime**
Chrome terminates MV3 service workers after ~30 seconds of inactivity. The model sessions are re-initialized from the Cache API on the next incoming message (typically 2–5 seconds).

## License

MIT
