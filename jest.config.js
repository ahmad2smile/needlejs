/** @type {import('jest').Config} */
export default {
  testEnvironment: "node",
  extensionsToTreatAsEsm: [".ts"],
  transform: {
    "^.+\\.ts$": ["ts-jest", {
      useESM: true,
      tsconfig: {
        module: "ESNext",
        target: "ES2020",
        moduleResolution: "node",
        esModuleInterop: true,
        allowJs: true,
        noImplicitAny: false,
        strict: false,
      },
    }],
  },
  moduleNameMapper: {
    "^(\\.{1,2}/.*)\\.js$": "$1",
  },
  testMatch: ["**/test/**/*.test.ts"],
};
