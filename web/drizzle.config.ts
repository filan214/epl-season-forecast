import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";

// Local tooling (generate / migrate / studio) reads the Neon URL from
// .env.local. In GitHub Actions the same value comes from the environment
// (secret NEON_DATABASE_URL, exported as DATABASE_URL). `generate` does not
// need a live connection; only `migrate`/`studio` do.
config({ path: ".env.local" });

export default defineConfig({
  schema: "./src/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL ?? "",
  },
  strict: true,
  verbose: true,
});
