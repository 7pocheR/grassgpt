#!/bin/bash

# Setup automatic continuation chains for TIER 1 experiments

echo "=== Setting up Tier 1 Continuation Chains ==="
echo ""

# Chain 1: 36L r=384 hybrid (needs ~55 sessions for 168k iters)
echo "Chain 1: 36L, r=384, hybrid (55 sessions)"
PREV=610968
for i in {1..55}; do
  NEXT=$(sbatch --dependency=afterany:$PREV --parsable submit_tier1_36L_r384_continue.sh)
  if [ $((i % 10)) -eq 0 ]; then
    echo "  Session $i: $NEXT"
  fi
  PREV=$NEXT
done
echo "  Final session 55: $PREV"
echo ""

# Chain 2: 24L r=512 hybrid (needs ~10 sessions for 100k iters)
echo "Chain 2: 24L, r=512, hybrid (10 sessions)"
PREV=610969
for i in {1..10}; do
  NEXT=$(sbatch --dependency=afterany:$PREV --parsable submit_tier1_24L_r512_continue.sh)
  echo "  Session $i: $NEXT"
  PREV=$NEXT
done
echo ""

# Chain 3: 24L r=384 full block (needs ~10 sessions for 100k iters)
echo "Chain 3: 24L, r=384, full block (10 sessions)"
PREV=610970
for i in {1..10}; do
  NEXT=$(sbatch --dependency=afterany:$PREV --parsable submit_tier1_24L_fullblock_continue.sh)
  echo "  Session $i: $NEXT"
  PREV=$NEXT
done
echo ""

echo "=== All continuation chains submitted ==="
echo "Total: 75 continuation sessions queued"
