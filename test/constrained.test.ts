/**
 * Unit tests for constrained decoding.
 */

import {
  Trie,
  ToolConstraints,
  JsonStateMachine,
  JsonState,
  ConstrainedDecoder,
  applyConstraints,
  TokenIndex,
} from '../src/constrained.js';

// ---------------------------------------------------------------------------
// Trie tests
// ---------------------------------------------------------------------------

describe('Trie', () => {
  test('insert and getNode', () => {
    const trie = new Trie();
    trie.insert('get_weather');
    trie.insert('send_email');

    expect(trie.getNode('get')).not.toBeNull();
    expect(trie.getNode('get_weather')).not.toBeNull();
    expect(trie.getNode('get_weather').isTerminal).toBe(true);
    expect(trie.getNode('get_weath').isTerminal).toBe(false);
    expect(trie.getNode('unknown')).toBeNull();
  });

  test('words returns all inserted words', () => {
    const trie = new Trie();
    trie.insert('foo');
    trie.insert('bar');
    trie.insert('baz');
    expect(trie.words.sort()).toEqual(['bar', 'baz', 'foo']);
  });
});

// ---------------------------------------------------------------------------
// ToolConstraints tests
// ---------------------------------------------------------------------------

const TOOLS_JSON = JSON.stringify([
  {
    name: 'get_weather',
    description: 'Get weather',
    parameters: {
      location: { type: 'string', description: 'City' },
      units: { type: 'string', description: 'Units' },
    },
  },
  {
    name: 'send_email',
    description: 'Send email',
    parameters: {
      to: { type: 'string', description: 'Recipient' },
      body: { type: 'string', description: 'Body' },
    },
  },
]);

describe('ToolConstraints', () => {
  test('name trie contains all tool names', () => {
    const tc = new ToolConstraints(TOOLS_JSON);
    expect(tc.nameTrie.getNode('get_weather').isTerminal).toBe(true);
    expect(tc.nameTrie.getNode('send_email').isTerminal).toBe(true);
    expect(tc.nameTrie.getNode('unknown')).toBeNull();
  });

  test('param trie contains parameter names for each tool', () => {
    const tc = new ToolConstraints(TOOLS_JSON);
    const pt = tc.getParamTrie('get_weather');
    expect(pt).not.toBeNull();
    expect(pt.getNode('location').isTerminal).toBe(true);
    expect(pt.getNode('units').isTerminal).toBe(true);
    expect(pt.getNode('unknown')).toBeNull();
  });

  test('handles invalid JSON gracefully', () => {
    const tc = new ToolConstraints('{invalid}');
    expect(tc.nameTrie.words).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// JsonStateMachine tests
// ---------------------------------------------------------------------------

describe('JsonStateMachine', () => {
  test('transitions to IN_NAME after "name":"', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"');
    expect(m.state).toBe(JsonState.IN_NAME);
    expect(m.constrainedBuf).toBe('');
  });

  test('captures tool name and returns to FREE after closing quote', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"get_weather"');
    expect(m.state).toBe(JsonState.FREE);
    expect(m.currentFunction).toBe('get_weather');
  });

  test('transitions to IN_ARG_KEY after entering arguments', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"get_weather","arguments":{"');
    expect(m.state).toBe(JsonState.IN_ARG_KEY);
    expect(m.constrainedBuf).toBe('');
  });

  test('captures argument key and returns to FREE', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"get_weather","arguments":{"location"');
    expect(m.state).toBe(JsonState.FREE);
  });

  test('stays FREE after value string', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"get_weather","arguments":{"location":"San Francisco"');
    expect(m.state).toBe(JsonState.FREE);
  });

  test('IN_ARG_KEY again for second key', () => {
    const m = new JsonStateMachine();
    m.feed('[{"name":"get_weather","arguments":{"location":"SF","');
    expect(m.state).toBe(JsonState.IN_ARG_KEY);
  });
});

// ---------------------------------------------------------------------------
// applyConstraints tests
// ---------------------------------------------------------------------------

describe('applyConstraints', () => {
  const VOCAB = 10;

  // Token 0 -> '', 1 -> 'g', 2 -> 'ge', 3 -> 'get', 4 -> 'get_', 5 -> 'get_w', 6 -> '"', 7 -> 'x', 8 -> ' ', 9 -> 'get_weather"'
  const tokenStrings = ['', 'g', 'ge', 'get', 'get_', 'get_w', '"', 'x', ' ', 'get_weather"'];
  const tokenIndex = new TokenIndex(tokenStrings);

  test('allows valid prefix tokens and terminal-closing token', () => {
    const trie = new Trie();
    trie.insert('get_weather');

    const logits = new Float32Array(VOCAB).fill(1.0);
    const node = trie.getNode('');
    const result = applyConstraints(logits, node, tokenStrings, tokenIndex);

    // 'g' starts a valid path
    expect(result[1]).toBe(1.0);
    // 'x' is not a valid start
    expect(result[7]).toBe(-Infinity);
    // ' ' (space) is not valid
    expect(result[8]).toBe(-Infinity);
    // 'get_weather"' completes the name
    expect(result[9]).toBe(1.0);
  });

  test('allows closing quote when at terminal node', () => {
    const trie = new Trie();
    trie.insert('get_weather');

    const logits = new Float32Array(VOCAB).fill(1.0);
    const node = trie.getNode('get_weather');
    expect(node.isTerminal).toBe(true);
    const result = applyConstraints(logits, node, tokenStrings, tokenIndex);

    // '"' is valid (closes the name)
    expect(result[6]).toBe(1.0);
    // 'g' would be off-trie from here
    expect(result[1]).toBe(-Infinity);
  });
});
