CREATE TYPE "public"."match_status" AS ENUM('scheduled', 'finished');--> statement-breakpoint
CREATE TYPE "public"."model_variant" AS ENUM('bayes', 'xgb', 'blend', 'market');--> statement-breakpoint
CREATE TYPE "public"."run_status" AS ENUM('ok', 'gated', 'failed');--> statement-breakpoint
CREATE TABLE "calibration_metrics" (
	"id" serial PRIMARY KEY NOT NULL,
	"model_run_id" integer NOT NULL,
	"through_gameweek" integer NOT NULL,
	"model_variant" "model_variant" NOT NULL,
	"n_matches" integer,
	"rps" double precision,
	"log_loss" double precision,
	"brier" double precision
);
--> statement-breakpoint
CREATE TABLE "matches" (
	"id" serial PRIMARY KEY NOT NULL,
	"season" text NOT NULL,
	"gameweek" integer NOT NULL,
	"kickoff_utc" timestamp with time zone NOT NULL,
	"home_team_id" integer NOT NULL,
	"away_team_id" integer NOT NULL,
	"home_goals" integer,
	"away_goals" integer,
	"status" "match_status" DEFAULT 'scheduled' NOT NULL,
	"source_ref" text
);
--> statement-breakpoint
CREATE TABLE "model_runs" (
	"id" serial PRIMARY KEY NOT NULL,
	"run_started_at" timestamp with time zone DEFAULT now() NOT NULL,
	"cutoff_date" date,
	"git_sha" text,
	"half_life_days" double precision,
	"n_draws" integer,
	"blend_weight" double precision,
	"r_hat_max" double precision,
	"divergences" integer,
	"ess_bulk_min" double precision,
	"status" "run_status" NOT NULL,
	"notes" text
);
--> statement-breakpoint
CREATE TABLE "predictions" (
	"id" serial PRIMARY KEY NOT NULL,
	"model_run_id" integer NOT NULL,
	"match_id" integer NOT NULL,
	"model_variant" "model_variant" NOT NULL,
	"p_home" double precision,
	"p_draw" double precision,
	"p_away" double precision,
	"lambda_home" double precision,
	"lambda_away" double precision,
	"p_over_2_5" double precision,
	"p_btts" double precision,
	"p_home_cs" double precision,
	"p_away_cs" double precision,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "season_sim_snapshots" (
	"id" serial PRIMARY KEY NOT NULL,
	"model_run_id" integer NOT NULL,
	"gameweek" integer NOT NULL,
	"team_id" integer NOT NULL,
	"p_title" double precision,
	"p_top4" double precision,
	"p_top6" double precision,
	"p_relegation" double precision,
	"exp_points" double precision,
	"points_p10" double precision,
	"points_p90" double precision,
	"exp_position" double precision,
	"with_param_uncertainty" boolean NOT NULL
);
--> statement-breakpoint
CREATE TABLE "team_strengths" (
	"id" serial PRIMARY KEY NOT NULL,
	"model_run_id" integer NOT NULL,
	"team_id" integer NOT NULL,
	"att_mean" double precision,
	"att_hdi_low" double precision,
	"att_hdi_high" double precision,
	"def_mean" double precision,
	"def_hdi_low" double precision,
	"def_hdi_high" double precision
);
--> statement-breakpoint
CREATE TABLE "teams" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"short_name" text,
	"slug" text NOT NULL,
	"fd_name" text,
	"understat_name" text,
	"transfermarkt_id" text,
	"is_promoted" boolean DEFAULT false NOT NULL,
	CONSTRAINT "teams_slug_unique" UNIQUE("slug")
);
--> statement-breakpoint
ALTER TABLE "calibration_metrics" ADD CONSTRAINT "calibration_metrics_model_run_id_model_runs_id_fk" FOREIGN KEY ("model_run_id") REFERENCES "public"."model_runs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "matches" ADD CONSTRAINT "matches_home_team_id_teams_id_fk" FOREIGN KEY ("home_team_id") REFERENCES "public"."teams"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "matches" ADD CONSTRAINT "matches_away_team_id_teams_id_fk" FOREIGN KEY ("away_team_id") REFERENCES "public"."teams"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "predictions" ADD CONSTRAINT "predictions_model_run_id_model_runs_id_fk" FOREIGN KEY ("model_run_id") REFERENCES "public"."model_runs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "predictions" ADD CONSTRAINT "predictions_match_id_matches_id_fk" FOREIGN KEY ("match_id") REFERENCES "public"."matches"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "season_sim_snapshots" ADD CONSTRAINT "season_sim_snapshots_model_run_id_model_runs_id_fk" FOREIGN KEY ("model_run_id") REFERENCES "public"."model_runs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "season_sim_snapshots" ADD CONSTRAINT "season_sim_snapshots_team_id_teams_id_fk" FOREIGN KEY ("team_id") REFERENCES "public"."teams"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "team_strengths" ADD CONSTRAINT "team_strengths_model_run_id_model_runs_id_fk" FOREIGN KEY ("model_run_id") REFERENCES "public"."model_runs"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "team_strengths" ADD CONSTRAINT "team_strengths_team_id_teams_id_fk" FOREIGN KEY ("team_id") REFERENCES "public"."teams"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "calibration_metrics_gw_variant_idx" ON "calibration_metrics" USING btree ("through_gameweek","model_variant");--> statement-breakpoint
CREATE INDEX "matches_gameweek_idx" ON "matches" USING btree ("gameweek");--> statement-breakpoint
CREATE INDEX "predictions_match_variant_idx" ON "predictions" USING btree ("match_id","model_variant");--> statement-breakpoint
CREATE INDEX "season_sim_snapshots_gw_team_idx" ON "season_sim_snapshots" USING btree ("gameweek","team_id");