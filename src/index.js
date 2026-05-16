/**
 * NeedleJS — Needle tool-calling model running locally via ONNX.
 *
 * The model files (encoder, decoder, tokenizer) ship inside the package under
 * models/, so the no-arg form of Needle.load() just works:
 *
 *   import { Needle } from 'needlejs';
 *   const needle = new Needle();
 *   await needle.load();
 *   const result = await needle.generate('What is the weather in SF?', tools);
 */

import { fileURLToPath } from 'url';
import path from 'path';

export { NeedleTokenizer, normalizeTools, restoreToolNames, toSnakeCase } from './tokenizer.js';
export { NeedleModel, configureOrtWasm } from './model.js';
export { generate } from './generator.js';
export { buildConstrainedDecoder, ConstrainedDecoder, JsonStateMachine, Trie, ToolConstraints } from './constrained.js';

import { NeedleTokenizer } from './tokenizer.js';
import { NeedleModel } from './model.js';
import { generate as _generate } from './generator.js';

/**
 * Resolve the paths to the model files bundled with this package. Works whether
 * the module is loaded from src/ (dev) or dist/ (built) — both sit one level
 * under the package root.
 */
export function bundledModelPaths() {
  const here = fileURLToPath(import.meta.url);
  const root = path.dirname(path.dirname(here));
  return {
    encoderPath: path.join(root, 'models', 'needle_encoder_fp16.onnx'),
    decoderPath: path.join(root, 'models', 'needle_decoder_fp16.onnx'),
    tokenizerPath: path.join(root, 'models', 'tokenizer', 'needle.model'),
    vocabPath: path.join(root, 'models', 'tokenizer', 'needle.vocab'),
  };
}

/**
 * High-level facade combining tokenizer, model, and generation.
 *
 * `load()` with no options uses the model files bundled in the package; pass
 * individual paths to override.
 */
export class Needle {
  constructor() {
    this.model = new NeedleModel();
    this.tokenizer = null;
  }

  /**
   * Load encoder, decoder and tokenizer.
   *
   * @param {object} [opts]
   * @param {string} [opts.encoderPath] - Path to encoder .onnx (defaults to bundled)
   * @param {string} [opts.decoderPath] - Path to decoder .onnx (defaults to bundled)
   * @param {string} [opts.tokenizerPath] - Path to tokenizer .model (defaults to bundled)
   * @param {string} [opts.vocabPath] - Path to tokenizer .vocab (defaults to alongside tokenizerPath)
   * @param {boolean} [opts.useWebGPU=false]
   */
  async load(opts = {}) {
    const def = bundledModelPaths();
    const {
      encoderPath = def.encoderPath,
      decoderPath = def.decoderPath,
      tokenizerPath = def.tokenizerPath,
      vocabPath = def.vocabPath,
      useWebGPU = false,
    } = opts;

    const [, tok] = await Promise.all([
      this.model.loadFromPaths(encoderPath, decoderPath, { useWebGPU }),
      NeedleTokenizer.fromPath(tokenizerPath, vocabPath),
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
