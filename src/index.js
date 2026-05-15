/**
 * NeedleJS — Needle tool-calling model running in-browser via ONNX.js.
 *
 * Quick start:
 *   import { Needle } from 'needlejs';
 *   const needle = new Needle();
 *   await needle.load({
 *     encoderUrl: 'https://your-cdn.example.com/needle_encoder_fp16.onnx',
 *     decoderUrl: 'https://your-cdn.example.com/needle_decoder_fp16.onnx',
 *     tokenizerUrl: 'https://your-cdn.example.com/tokenizer/needle.model',
 *     wasmDir: 'https://your-cdn.example.com/ort-wasm/',
 *   });
 *   const result = await needle.generate('What is the weather in SF?', tools);
 */

export { NeedleTokenizer, normalizeTools, restoreToolNames, toSnakeCase } from './tokenizer.js';
export { NeedleModel, configureOrtWasm } from './model.js';
export { generate } from './generator.js';
export { buildConstrainedDecoder, ConstrainedDecoder, JsonStateMachine, Trie, ToolConstraints } from './constrained.js';

import { NeedleTokenizer } from './tokenizer.js';
import { NeedleModel, configureOrtWasm } from './model.js';
import { generate as _generate } from './generator.js';

/**
 * High-level facade combining tokenizer, model, and generation.
 */
export class Needle {
  constructor() {
    this.model = new NeedleModel();
    this.tokenizer = null;
  }

  /**
   * Load all model components.
   * @param {object} opts
   * @param {string} opts.encoderUrl
   * @param {string} opts.decoderUrl
   * @param {string} opts.tokenizerUrl
   * @param {string} [opts.wasmDir] - defaults to same dir as encoderUrl
   * @param {boolean} [opts.useWebGPU=false]
   */
  async load({ encoderUrl, decoderUrl, tokenizerUrl, wasmDir, useWebGPU = false }) {
    if (wasmDir) configureOrtWasm(wasmDir);
    const [, tok] = await Promise.all([
      this.model.loadFromUrls(encoderUrl, decoderUrl, { useWebGPU }),
      NeedleTokenizer.fromUrl(tokenizerUrl),
    ]);
    this.tokenizer = tok;
  }

  /**
   * Generate a tool-call for the given query and tool definitions.
   *
   * @param {string} query
   * @param {string|object[]} tools - JSON string or array of tool definitions
   * @param {object} [opts] - passed through to generate()
   * @returns {Promise<string>} Tool-call JSON string
   */
  async generate(query, tools, opts = {}) {
    if (!this.tokenizer || !this.model.isLoaded) {
      throw new Error('Call needle.load() before generate()');
    }
    return _generate(this.model, this.tokenizer, query, tools, opts);
  }
}
