/**
 * Rollup config for the Chrome extension bundle.
 * Bundles everything (including onnxruntime-web) into a single background.js.
 */

import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import copy from 'rollup-plugin-copy';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default {
  input: 'extension/background.js',
  output: {
    file: 'dist/extension/background.js',
    format: 'esm',
    sourcemap: false,
  },
  plugins: [
    resolve({ browser: true, preferBuiltins: false }),
    commonjs(),
    copy({
      targets: [
        // Extension manifest and popup
        { src: 'extension/manifest.json', dest: 'dist/extension' },
        { src: 'extension/popup', dest: 'dist/extension' },
        // ORT WASM files
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'dist/extension/ort-wasm',
        },
        {
          src: 'node_modules/onnxruntime-web/dist/ort-wasm*.mjs',
          dest: 'dist/extension/ort-wasm',
        },
        // Models (if present locally)
        {
          src: 'models/*.onnx',
          dest: 'dist/extension/models',
          errorOnMissing: false,
        },
        {
          src: 'models/tokenizer/needle.model',
          dest: 'dist/extension/models/tokenizer',
          errorOnMissing: false,
        },
      ],
    }),
  ],
};
