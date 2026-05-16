import { Needle } from "../src/index";

describe("Needle Flow", () => {
  test("end-to-end tool call uses bundled models", async () => {
    const needle = new Needle();
    await needle.load();

    const tools = [
      {
        name: "get_weather",
        description: "Get current weather for a city.",
        parameters: {
          location: { type: "string", description: "City name." },
        },
      },
      {
        name: "get_whatever",
        description: "Get current whatever for a city.",
        parameters: {
          location: { type: "string", description: "City name." },
        },
      },
    ];

    const result = await needle.generate("Weather of San Francisco?", tools);

    expect(result).toBe(
      '[{"name":"get_weather","arguments":{"location":"San Francisco"}}]',
    );
  }, 120_000);
});
