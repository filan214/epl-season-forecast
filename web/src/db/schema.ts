// web/src/db/schema.ts
//
// Drizzle is the single schema authority for the whole project
// (IMPLEMENTATION_GUIDE.md §1 & §3). The Python pipeline writes through raw
// parameterised SQL against the tables this file creates; it never owns schema.
// One schema authority, two consumers.
//
// Seven tables — teams, matches, model_runs, predictions, season_sim_snapshots,
// team_strengths, calibration_metrics — plus the four indexes from §3.
//
// LEAKAGE GUARD (project-critical): the single most important constraint — that
// every prediction row is created strictly BEFORE the referenced match kicks off
// — cannot be expressed in Drizzle, because a Postgres CHECK cannot reference
// another table. It is enforced by a BEFORE INSERT trigger defined in the raw
// SQL migration `drizzle/0001_leakage_guard.sql`. See PRD.md §6.5/§13 and
// IMPLEMENTATION_GUIDE.md §3 (predictions) / §10.

import {
  boolean,
  date,
  doublePrecision,
  index,
  integer,
  pgEnum,
  pgTable,
  serial,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

// ---- Enums -----------------------------------------------------------------

export const matchStatus = pgEnum("match_status", ["scheduled", "finished"]);

export const runStatus = pgEnum("run_status", ["ok", "gated", "failed"]);

// Model variants share one enum so `market` rows live in the same tables as the
// model rows (predictions, calibration_metrics). Every comparison then becomes a
// GROUP BY instead of a join across systems (IMPLEMENTATION_GUIDE.md §3).
export const modelVariant = pgEnum("model_variant", [
  "bayes",
  "xgb",
  "blend",
  "market",
]);

// ---- teams -----------------------------------------------------------------
// Name reconciliation across all four data sources. Every ingest module resolves
// through this table; a silent join failure here corrupts everything downstream.
export const teams = pgTable("teams", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  shortName: text("short_name"),
  slug: text("slug").notNull().unique(), // used in /team/[slug]
  fdName: text("fd_name"), // football-data.co.uk spelling
  understatName: text("understat_name"),
  transfermarktId: text("transfermarkt_id"),
  isPromoted: boolean("is_promoted").notNull().default(false), // for 2026-27
});

// ---- matches ---------------------------------------------------------------
export const matches = pgTable(
  "matches",
  {
    id: serial("id").primaryKey(),
    season: text("season").notNull(),
    gameweek: integer("gameweek").notNull(),
    kickoffUtc: timestamp("kickoff_utc", { withTimezone: true }).notNull(),
    homeTeamId: integer("home_team_id")
      .notNull()
      .references(() => teams.id),
    awayTeamId: integer("away_team_id")
      .notNull()
      .references(() => teams.id),
    homeGoals: integer("home_goals"),
    awayGoals: integer("away_goals"),
    status: matchStatus("status").notNull().default("scheduled"),
    sourceRef: text("source_ref"),
  },
  (t) => [index("matches_gameweek_idx").on(t.gameweek)],
);

// ---- model_runs ------------------------------------------------------------
// Provenance root: every other table points back here, tracing a set of numbers
// to a git SHA and a cutoff date. A gated run writes ONLY this row (status).
export const modelRuns = pgTable("model_runs", {
  id: serial("id").primaryKey(),
  runStartedAt: timestamp("run_started_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  cutoffDate: date("cutoff_date"),
  gitSha: text("git_sha"),
  halfLifeDays: doublePrecision("half_life_days"),
  nDraws: integer("n_draws"),
  blendWeight: doublePrecision("blend_weight"),
  rHatMax: doublePrecision("r_hat_max"),
  divergences: integer("divergences"),
  essBulkMin: doublePrecision("ess_bulk_min"),
  status: runStatus("status").notNull(),
  notes: text("notes"),
});

// ---- predictions -----------------------------------------------------------
// created_at < matches.kickoff_utc is enforced by the BEFORE INSERT trigger in
// drizzle/0001_leakage_guard.sql (Drizzle cannot express a cross-table CHECK).
// The `market` variant is stored here too — same table, same shape.
export const predictions = pgTable(
  "predictions",
  {
    id: serial("id").primaryKey(),
    modelRunId: integer("model_run_id")
      .notNull()
      .references(() => modelRuns.id),
    matchId: integer("match_id")
      .notNull()
      .references(() => matches.id),
    modelVariant: modelVariant("model_variant").notNull(),
    pHome: doublePrecision("p_home"),
    pDraw: doublePrecision("p_draw"),
    pAway: doublePrecision("p_away"),
    lambdaHome: doublePrecision("lambda_home"),
    lambdaAway: doublePrecision("lambda_away"),
    pOver25: doublePrecision("p_over_2_5"),
    pBtts: doublePrecision("p_btts"),
    pHomeCs: doublePrecision("p_home_cs"),
    pAwayCs: doublePrecision("p_away_cs"),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (t) => [index("predictions_match_variant_idx").on(t.matchId, t.modelVariant)],
);

// ---- season_sim_snapshots --------------------------------------------------
// Both uncertainty variants are written each run (with_param_uncertainty),
// powering the side-by-side exhibit in PRD §6.3.
export const seasonSimSnapshots = pgTable(
  "season_sim_snapshots",
  {
    id: serial("id").primaryKey(),
    modelRunId: integer("model_run_id")
      .notNull()
      .references(() => modelRuns.id),
    gameweek: integer("gameweek").notNull(),
    teamId: integer("team_id")
      .notNull()
      .references(() => teams.id),
    pTitle: doublePrecision("p_title"),
    pTop4: doublePrecision("p_top4"),
    pTop6: doublePrecision("p_top6"),
    pRelegation: doublePrecision("p_relegation"),
    expPoints: doublePrecision("exp_points"),
    pointsP10: doublePrecision("points_p10"),
    pointsP90: doublePrecision("points_p90"),
    expPosition: doublePrecision("exp_position"),
    withParamUncertainty: boolean("with_param_uncertainty").notNull(),
  },
  (t) => [index("season_sim_snapshots_gw_team_idx").on(t.gameweek, t.teamId)],
);

// ---- team_strengths --------------------------------------------------------
// `def_mean` is signed so that HIGHER means BETTER defence — the plotting code
// asserts this (IMPLEMENTATION_GUIDE.md §10).
export const teamStrengths = pgTable("team_strengths", {
  id: serial("id").primaryKey(),
  modelRunId: integer("model_run_id")
    .notNull()
    .references(() => modelRuns.id),
  teamId: integer("team_id")
    .notNull()
    .references(() => teams.id),
  attMean: doublePrecision("att_mean"),
  attHdiLow: doublePrecision("att_hdi_low"),
  attHdiHigh: doublePrecision("att_hdi_high"),
  defMean: doublePrecision("def_mean"),
  defHdiLow: doublePrecision("def_hdi_low"),
  defHdiHigh: doublePrecision("def_hdi_high"),
});

// ---- calibration_metrics ---------------------------------------------------
// Market rows use model_variant = 'market', so the comparison chart is one query.
export const calibrationMetrics = pgTable(
  "calibration_metrics",
  {
    id: serial("id").primaryKey(),
    modelRunId: integer("model_run_id")
      .notNull()
      .references(() => modelRuns.id),
    throughGameweek: integer("through_gameweek").notNull(),
    modelVariant: modelVariant("model_variant").notNull(),
    nMatches: integer("n_matches"),
    rps: doublePrecision("rps"),
    logLoss: doublePrecision("log_loss"),
    brier: doublePrecision("brier"),
  },
  (t) => [
    index("calibration_metrics_gw_variant_idx").on(
      t.throughGameweek,
      t.modelVariant,
    ),
  ],
);
