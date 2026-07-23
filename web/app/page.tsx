// Placeholder landing page. The real dashboard (table + probabilities,
// /calibration, /team/[slug], /methodology) is built in Phase 5
// (IMPLEMENTATION_GUIDE.md §6). This scaffold intentionally computes nothing.

export default function Home() {
  return (
    <main className="flex flex-1 items-center justify-center px-6">
      <div className="max-w-xl space-y-4 text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-muted-foreground">
          Scaffold
        </p>
        <h1 className="text-4xl font-semibold tracking-tight text-balance">
          EPL Season Forecast — 2026-27
        </h1>
        <p className="text-muted-foreground text-pretty">
          A live-updating Bayesian forecast of the Premier League season, with a
          public calibration record against bookmaker closing odds. The
          dashboard lands in Phase 5; this page is a placeholder.
        </p>
      </div>
    </main>
  );
}
