-- Verification for the leakage guard (IMPLEMENTATION_GUIDE.md §8, checkpoint P0).
--
-- Run against a database that already has the migrations applied, e.g.:
--   psql "$DATABASE_URL" -f web/scripts/verify_leakage_guard.sql
--
-- Everything runs inside a transaction that is ROLLED BACK, so it leaves no
-- data behind. Expected NOTICE output:
--   OK  : pre-kickoff prediction accepted.
--   OK  : post-kickoff prediction rejected (...).
--   PASS: leakage guard verified.
-- If the guard is missing/broken you get instead:
--   FAIL: leakage guard did NOT reject a post-kickoff prediction!

BEGIN;

DO $$
DECLARE
  v_home  int;
  v_away  int;
  v_run   int;
  v_match int;
  v_leaked boolean := false;
BEGIN
  INSERT INTO teams (name, slug) VALUES ('Guard Test Home', 'guard-test-home') RETURNING id INTO v_home;
  INSERT INTO teams (name, slug) VALUES ('Guard Test Away', 'guard-test-away') RETURNING id INTO v_away;
  INSERT INTO model_runs (status) VALUES ('ok') RETURNING id INTO v_run;
  INSERT INTO matches (season, gameweek, kickoff_utc, home_team_id, away_team_id)
    VALUES ('2026-27', 1, TIMESTAMPTZ '2026-08-15 14:00:00+00', v_home, v_away)
    RETURNING id INTO v_match;

  -- (1) created_at strictly before kickoff -> must be ACCEPTED.
  INSERT INTO predictions (model_run_id, match_id, model_variant, created_at)
    VALUES (v_run, v_match, 'blend', TIMESTAMPTZ '2026-08-14 09:00:00+00');
  RAISE NOTICE 'OK  : pre-kickoff prediction accepted.';

  -- (2) created_at at kickoff -> must be REJECTED by trg_predictions_no_leakage.
  --     A nested block catches the expected error so the check can continue.
  BEGIN
    INSERT INTO predictions (model_run_id, match_id, model_variant, created_at)
      VALUES (v_run, v_match, 'blend', TIMESTAMPTZ '2026-08-15 14:00:00+00');
    v_leaked := true; -- reached only if the trigger FAILED to fire
  EXCEPTION WHEN others THEN
    RAISE NOTICE 'OK  : post-kickoff prediction rejected (%).', SQLERRM;
  END;

  IF v_leaked THEN
    RAISE EXCEPTION 'FAIL: leakage guard did NOT reject a post-kickoff prediction!';
  END IF;

  RAISE NOTICE 'PASS: leakage guard verified.';
END $$;

ROLLBACK;
