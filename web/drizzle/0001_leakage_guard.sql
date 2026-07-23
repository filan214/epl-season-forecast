-- ============================================================================
-- LEAKAGE GUARD  --  the project's single most important database constraint.
-- ============================================================================
-- Enforces that every prediction row is written STRICTLY BEFORE the referenced
-- match kicks off (created_at < matches.kickoff_utc). This makes the core
-- integrity claim of the forecast -- that no prediction ever saw information
-- from at/after kickoff -- unfalsifiable by accident, at the database level
-- rather than trusting application code.
--
-- Why a trigger and not a CHECK: a Postgres CHECK constraint cannot reference a
-- column in another table, and the kickoff time lives on `matches`. So the rule
-- is implemented as a BEFORE INSERT trigger that raises on violation. Drizzle
-- cannot express this, which is why it lives in this hand-written migration
-- alongside the generated schema migration.
--
-- References: PRD.md §6.5 (operational requirements), §13 (risks / leakage),
-- IMPLEMENTATION_GUIDE.md §3 (predictions) and §10 (guardrails).
--
-- Column defaults (predictions.created_at DEFAULT now()) are resolved BEFORE
-- BEFORE-INSERT triggers fire, so NEW.created_at holds the real value here even
-- when the caller omits it.
-- ============================================================================

CREATE OR REPLACE FUNCTION check_prediction_predates_kickoff()
RETURNS TRIGGER AS $$
DECLARE
  match_kickoff timestamptz;
BEGIN
  SELECT kickoff_utc INTO match_kickoff
  FROM matches
  WHERE id = NEW.match_id;

  IF match_kickoff IS NULL THEN
    RAISE EXCEPTION
      'Leakage guard: match % has no kickoff_utc or does not exist', NEW.match_id;
  END IF;

  IF NEW.created_at >= match_kickoff THEN
    RAISE EXCEPTION
      'Leakage guard violation: prediction created_at (%) is not before match % kickoff_utc (%)',
      NEW.created_at, NEW.match_id, match_kickoff;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
--> statement-breakpoint
DROP TRIGGER IF EXISTS trg_predictions_no_leakage ON predictions;
--> statement-breakpoint
CREATE TRIGGER trg_predictions_no_leakage
  BEFORE INSERT ON predictions
  FOR EACH ROW
  EXECUTE FUNCTION check_prediction_predates_kickoff();
