import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';

export default [
  // ESM build (for bundlers)
  {
    input: 'src/index.js',
    output: {
      file: 'dist/needle.esm.js',
      format: 'esm',
      sourcemap: true,
    },
    plugins: [
      resolve({ browser: true }),
      commonjs(),
    ],
    external: ['onnxruntime-web', 'sentencepiece-js'],
  },
  // UMD build (for direct script include)
  {
    input: 'src/index.js',
    output: {
      file: 'dist/needle.umd.cjs',
      format: 'umd',
      name: 'NeedleJS',
      sourcemap: true,
      globals: {
        'onnxruntime-web': 'ort',
        'sentencepiece-js': 'SentencePieceJS',
      },
    },
    plugins: [
      resolve({ browser: true }),
      commonjs(),
    ],
    external: ['onnxruntime-web', 'sentencepiece-js'],
  },
];
