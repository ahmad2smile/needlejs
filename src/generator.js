/**
 * Autoregressive generation loop for Needle.
 *
 * Port of needle/model/run.py generate().
 */

import {
  EOS_ID, TOOLS_ID, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN,
  normalizeTools, restoreToolNames,
} from './tokenizer.js';
import { buildConstrainedDecoder } from './constrained.js';

/**
 * Return the index of the maximum value in a Float32Array slice.
 * @param {Float32Array} arr
 * @param {number} offset - start index in arr
 * @param {number} length - number of elements (vocab_size)
 */
function argmax(arr, offset, length) {
  let best = offset;
  for (let i = offset + 1; i < offset + length; i++) {
    if (arr[i] > arr[best]) best = i;
  }
  return best - offset;
}

/**
 * Generate a tool-call JSON string for the given query and tools.
 *
 * @param {import('./model.js').NeedleModel} model
 * @param {import('./tokenizer.js').NeedleTokenizer} tokenizer
 * @param {string} query - Natural language query
 * @param {string|object[]} tools - Tool definitions as JSON string or array
 * @param {object} [opts]
 * @param {number} [opts.maxGenLen=256]
 * @param {number} [opts.maxEncLen=1024]
 * @param {boolean} [opts.constrained=true]
 * @param {function(string): void} [opts.onToken] - Called with each decoded token string
 * @returns {Promise<string>} Decoded tool-call JSON
 */
export async function generate(model, tokenizer, query, tools, opts = {}) {
  const {
    maxGenLen = DEFAULT_MAX_GEN_LEN,
    maxEncLen = DEFAULT_MAX_ENC_LEN,
    constrained = true,
    onToken = null,
  } = opts;

  const toolsStr = typeof tools === 'string' ? tools : JSON.stringify(tools);
  const { normalizedJson, nameMap } = normalizeTools(toolsStr);

  // Build encoder input: [query..., TOOLS_ID, tool_tokens...]
  const encIds = tokenizer.buildEncoderInput(query, normalizedJson, maxEncLen);
  const encoderInput = new Int32Array(encIds);

  // Run encoder once
  const encoderHidden = await model.encode(encoderInput);

  // Initialize decoder buffer with [EOS=1, PAD, PAD, ...]
  const decBuffer = new Int32Array(maxGenLen).fill(0);
  decBuffer[0] = EOS_ID;

  // Build constrained decoder
  const constrainedDec = constrained
    ? buildConstrainedDecoder([normalizedJson], tokenizer)
    : null;

  const generated = [];

  for (let i = 0; i < maxGenLen - 1; i++) {
    // Run decoder on full buffer up to current position
    const currentDec = decBuffer.slice(0, i + 1);
    const logitsTensor = await model.decode(currentDec, encoderHidden);

    // logitsTensor data: [1, dec_len, vocab_size] in row-major
    // We want the logits at position i
    const vocabSize = logitsTensor.dims[2];
    const logitsData = logitsTensor.data; // Float32Array
    const offset = i * vocabSize;
    const logitsSlice = logitsData.subarray(offset, offset + vocabSize);

    // Apply constrained decoding mask if active
    let logits = logitsSlice;
    if (constrainedDec && constrainedDec.isActive(0)) {
      logits = constrainedDec.constrainLogits(new Float32Array(logitsSlice), 0);
    }

    const nextToken = argmax(logits, 0, vocabSize);

    if (constrainedDec) constrainedDec.update(0, nextToken);

    if (nextToken === EOS_ID) break;

    generated.push(nextToken);
    decBuffer[i + 1] = nextToken;

    if (onToken) {
      const piece = tokenizer.decode([nextToken]);
      onToken(piece);
    }
  }

  let result = tokenizer.decode(generated);
  // Strip leading <tool_call> token text if present
  if (result.startsWith('<tool_call>')) result = result.slice('<tool_call>'.length);

  return restoreToolNames(result, nameMap);
}
