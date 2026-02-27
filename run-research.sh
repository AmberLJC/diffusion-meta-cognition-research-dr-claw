#!/bin/bash
# Dr. Claw Research Runner — called every 15 min by cron
# Runs the research agent, commits progress, sends report

set -e

REPO_DIR="/home/azureuser/research/diffusion-meta-cognition-research-dr-claw"
cd "$REPO_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
REPORT_FILE="reports/report-$(date -u +"%Y%m%d-%H%M").md"

echo "🔬 Dr. Claw Research Runner — $TIMESTAMP"

# Run the research agent
claude --dangerously-skip-permissions -p "
You are a research agent working on: META-COGNITION IN DIFFUSION LANGUAGE MODELS.

Working directory: $REPO_DIR

Your job every 15 minutes:
1. Read existing notes/ ideas/ reports/ to understand current progress
2. Do deep web searches (use WebSearch tool) on:
   - Diffusion language models (MDLM, Plaid, SEDD, etc.)
   - Meta-cognition / mechanistic interpretability in LLMs
   - Differences from autoregressive models
   - Unexplored intersections (uncertainty, planning, knowledge boundaries)
3. Identify the MOST PROMISING and NOVEL research direction nobody has pursued
4. Write detailed notes to notes/YYYY-MM-DD.md
5. Update ideas/prioritized-directions.md with your ranked list
6. Write a concise progress report to: $REPORT_FILE

Report format:
# Progress Report — $TIMESTAMP

## What I searched/read this session
## Key findings
## Most promising novel direction (ranked #1 right now)
## Why it's unexplored
## Proposed next experiment
## Open questions

Be specific, cite papers, URLs. Think like a PhD student aiming for a Nature paper.
" 2>&1 | tee /tmp/dr-claw-run.log

# Commit all changes
git add -A
git diff --cached --quiet || git commit -m "🔬 research update — $TIMESTAMP"
git push origin main 2>/dev/null || git push --set-upstream origin main

echo "✅ Committed and pushed."
