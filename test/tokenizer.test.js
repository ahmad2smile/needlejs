/**
 * Unit tests for tokenizer utilities (no actual SentencePiece model needed).
 */

import { toSnakeCase, normalizeTools, restoreToolNames } from '../src/tokenizer.js';

describe('toSnakeCase', () => {
  const cases = [
    ['getWeather', 'get_weather'],
    ['GetWeather', 'get_weather'],
    ['get_weather', 'get_weather'],
    ['sendEmail', 'send_email'],
    ['XMLParser', 'xml_parser'],
    ['getUserID', 'get_user_id'],
    ['dot.notation.name', 'dot_notation_name'],
    ['already_snake', 'already_snake'],
  ];

  test.each(cases)('toSnakeCase(%s) = %s', (input, expected) => {
    expect(toSnakeCase(input)).toBe(expected);
  });
});

describe('normalizeTools / restoreToolNames', () => {
  const tools = [
    { name: 'getWeather', description: 'Get weather', parameters: {} },
    { name: 'sendEmail', description: 'Send email', parameters: {} },
  ];

  test('normalizes tool names to snake_case', () => {
    const { normalizedJson, nameMap } = normalizeTools(JSON.stringify(tools));
    const parsed = JSON.parse(normalizedJson);
    expect(parsed[0].name).toBe('get_weather');
    expect(parsed[1].name).toBe('send_email');
    expect(nameMap['get_weather']).toBe('getWeather');
    expect(nameMap['send_email']).toBe('sendEmail');
  });

  test('restores original tool names from model output', () => {
    const { nameMap } = normalizeTools(JSON.stringify(tools));
    const modelOutput = '{"name":"get_weather","arguments":{"location":"SF"}}';
    const restored = restoreToolNames(modelOutput, nameMap);
    const parsed = JSON.parse(restored);
    expect(parsed.name).toBe('getWeather');
  });

  test('handles empty nameMap gracefully', () => {
    const text = '{"name":"get_weather","arguments":{}}';
    expect(restoreToolNames(text, {})).toBe(text);
  });

  test('handles invalid tools JSON gracefully', () => {
    const { normalizedJson, nameMap } = normalizeTools('{bad json}');
    expect(normalizedJson).toBe('{bad json}');
    expect(nameMap).toEqual({});
  });
});
