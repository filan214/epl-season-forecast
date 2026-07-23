// web/src/db/client.ts
//
// Read-only serving client for the dashboard. Uses the Neon serverless HTTP
// driver so each query is a stateless fetch and the connection closes
// immediately — this preserves Neon scale-to-zero and avoids burning free-tier
// compute hours on an idle pooled connection (IMPLEMENTATION_GUIDE.md §6).
//
// The dashboard NEVER computes anything: no model calls, no simulation, no
// aggregation beyond what SQL does. If it can't be read straight from a table,
// the pipeline should have written it (IMPLEMENTATION_GUIDE.md §0).

import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "./schema";

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error(
    "DATABASE_URL is not set. Copy web/.env.example to web/.env.local and set the Neon connection string.",
  );
}

const sql = neon(databaseUrl);

export const db = drizzle(sql, { schema });
export { schema };
