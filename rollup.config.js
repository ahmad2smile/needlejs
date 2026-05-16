import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

const external = ['onnxruntime-web', 'sentencepiece-js', 'url', 'path', 'fs'];

export default [
  // ESM build (for bundlers and Node)
  {
    input: 'src/index.js',
    output: {
      file: 'dist/needle.esm.js',
      format: 'esm',
      sourcemap: true,
    },
    plugins: [resolve({ preferBuiltins: true }), commonjs()],
    external,
  },
  // CommonJS build (for `require()`)
  {
    input: 'src/index.js',
    output: {
      file: 'dist/needle.umd.cjs',
      format: 'cjs',
      exports: 'named',
      sourcemap: true,
    },
    plugins: [resolve({ preferBuiltins: true }), commonjs()],
    external,
  },
];
