#!/usr/bin/env bash
# Train every leave-performer-out fold for one seed, then score them.
#
# This is the command that turns a config into a number. Leave-performer-out
# means the model is tested on people it has never seen, so a result needs all
# ten folds: a single fold is one group of nine performers and moves by about
# 2.7 points on its own, while the ten-fold mean is stable to a tenth.
#
# Resumable. A fold whose `last.ckpt` already exists is skipped, so if you
# interrupt this you can simply run it again. Scoring only happens once every
# fold is present, so a partial run never produces a misleading number.
#
# Usage:
#   scripts/run_lpo.sh                                  # the 7-emotion recipe, seed 255
#   SEED=1 scripts/run_lpo.sh                           # another seed
#   CONFIG=configs/diema13_stgcn_recipe.yaml scripts/run_lpo.sh
#   EXTRA="model.plusplus=true" scripts/run_lpo.sh      # the STGCN++ temporal block
#   UNTIL=08:00 scripts/run_lpo.sh                      # stop cleanly at a wall-clock time
#
# Run it detached if it will outlive your terminal:
#   nohup setsid scripts/run_lpo.sh > run.log 2>&1 &    # Linux
#   nohup scripts/run_lpo.sh > run.log 2>&1 &           # macOS, which has no setsid
#
# Cost: about 18 minutes per fold on one RTX 4090 for the 7-emotion recipe at
# 400 epochs, so roughly 3 hours per seed. The 13-label corpus is about twice
# that. Averaging three seeds is worth roughly three points and needs three
# runs of this script with SEED set differently.
set -uo pipefail

CONFIG=${CONFIG:-configs/diema7_stgcn_recipe.yaml}
SEED=${SEED:-255}
FOLDS=${FOLDS:-10}
PY=${PY:-python}
RUNS=${RUNS:-runs}
NAME=${NAME:-$(basename "$CONFIG" .yaml)}
EXTRA=${EXTRA:-}
UNTIL=${UNTIL:-}

LOGS="$RUNS/$NAME/seed$SEED"
OUT="$RUNS/$NAME/seed${SEED}.json"

# The next occurrence of UNTIL (HH:MM), as epoch seconds. Computed in Python
# rather than with `date -d`, which only GNU date has: on macOS it failed, the
# deadline stayed empty, and the run silently ignored UNTIL.
deadline=""
if [ -n "$UNTIL" ]; then
  deadline=$($PY - "$UNTIL" <<'PYEOF'
import datetime as dt, sys
at = dt.datetime.combine(dt.date.today(), dt.datetime.strptime(sys.argv[1], "%H:%M").time())
if at <= dt.datetime.now():
    at += dt.timedelta(days=1)
print(int(at.timestamp()))
PYEOF
  ) || { echo "!!! could not read a deadline from UNTIL=$UNTIL (expected HH:MM, e.g. 08:00)"; exit 2; }
fi
past_deadline() { [ -n "$deadline" ] && [ "$(date +%s)" -ge "$deadline" ]; }
fold_done() { compgen -G "$LOGS/recipe/fold$(printf %02d "$1")/*/version_*/checkpoints/last.ckpt" > /dev/null; }

mkdir -p "$LOGS"; fail=0
echo "=== $CONFIG | seed $SEED | $FOLDS folds | -> $OUT ==="
[ -n "$EXTRA" ] && echo "    extra overrides: $EXTRA"

for fold in $(seq 1 "$FOLDS"); do
  if fold_done "$fold"; then echo "--- fold $fold/$FOLDS already done, skipping ---"; continue; fi
  if past_deadline; then echo "=== reached UNTIL=$UNTIL, stopping cleanly before fold $fold ==="; exit "$fail"; fi
  echo "--- fold $fold/$FOLDS [$(date +%H:%M:%S)] ---"
  $PY -m emo_mocap.cli.train --config "$CONFIG" --fold "$fold" --num-folds "$FOLDS" \
      --override "data.seed=$SEED" \
        "logging.log_dir=$LOGS/recipe/fold{fold}/" \
        "checkpointing.save=[last]" "checkpointing.test_with=current" \
        $EXTRA \
    || { echo "!!! fold $fold FAILED"; fail=$((fail+1)); }
done

complete=1
for fold in $(seq 1 "$FOLDS"); do fold_done "$fold" || complete=0; done
if [ "$complete" -eq 0 ]; then
  echo "=== not every fold is present, so nothing was scored. Re-run to finish. ==="
  exit "$fail"
fi

echo "=== scoring $FOLDS folds ==="
$PY scripts/score_folds.py --config "$CONFIG" --logs "$LOGS" --variants recipe \
    --folds "$FOLDS" --out "$OUT" --override "data.seed=$SEED" $EXTRA \
  || { echo "!!! scoring FAILED"; fail=$((fail+1)); }

$PY - "$OUT" <<'PYEOF'
import json, sys, statistics as st
d = json.load(open(sys.argv[1])); v = next(iter(d))
folds = {int(k): rows[-1] for k, rows in d[v].items()}
def col(split): return [100 * folds[f][split]["acc"] for f in sorted(folds)]
t, va, tr = col("test"), col("val"), col("train")
print(f"\n  test  {st.mean(t):6.2f}%   (folds {min(t):.1f} to {max(t):.1f})")
print(f"  val   {st.mean(va):6.2f}%")
print(f"  train {st.mean(tr):6.2f}%   <- below ~99.9 means the run did not finish fitting")
print(f"\n  Averaging several seeds is worth about 4 points. See the README.\n")
PYEOF
exit "$fail"
