#!/usr/bin/env bash
# Project Gate — lightweight releasability check
# Runs: spec traceability check + tests
# Usage: bash scripts/gate.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPECS_DIR="$PROJECT_ROOT/specs"
TEST_DIR="$PROJECT_ROOT/test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

echo "============================================"
echo "  IWS_sciml Project Gate"
echo "============================================"
echo ""

# ------------------------------------------------
# Check 1: Spec traceability
# Every behavior ID in specs/ must have a matching @testset in test/
# Every behavior ID in test/ must exist in a spec file
# ------------------------------------------------
echo "--- Check 1: Spec Traceability ---"

# Extract behavior IDs from spec files
SPEC_IDS=$(grep -rh "id:" "$SPECS_DIR"/*.yaml 2>/dev/null | sed 's/.*id:\s*//' | tr -d ' ' | sort)
SPEC_COUNT=$(echo "$SPEC_IDS" | wc -l)

# Extract behavior IDs from test files (pattern: @testset "ID-NNN: ...")
TEST_IDS=$(grep -rh '@testset' "$TEST_DIR"/test_*.jl 2>/dev/null | grep -oP '[A-Z]+-\d+' | sort)
TEST_COUNT=$(echo "$TEST_IDS" | wc -l)

echo "  Spec behavior IDs: $SPEC_COUNT"
echo "  Test behavior IDs: $TEST_COUNT"

# Find spec IDs without tests
ORPHAN_SPECS=$(comm -23 <(echo "$SPEC_IDS") <(echo "$TEST_IDS"))
if [ -n "$ORPHAN_SPECS" ]; then
    echo -e "  ${RED}FAIL: Spec IDs without tests:${NC}"
    echo "$ORPHAN_SPECS" | while read -r id; do
        echo "    - $id"
    done
    FAIL=$((FAIL + 1))
else
    echo -e "  ${GREEN}PASS: All spec IDs have tests${NC}"
    PASS=$((PASS + 1))
fi

# Find test IDs without specs
ORPHAN_TESTS=$(comm -23 <(echo "$TEST_IDS") <(echo "$SPEC_IDS"))
if [ -n "$ORPHAN_TESTS" ]; then
    echo -e "  ${RED}FAIL: Test IDs without specs:${NC}"
    echo "$ORPHAN_TESTS" | while read -r id; do
        echo "    - $id"
    done
    FAIL=$((FAIL + 1))
else
    echo -e "  ${GREEN}PASS: All test IDs have specs${NC}"
    PASS=$((PASS + 1))
fi

echo ""

# ------------------------------------------------
# Check 2: No deleted/skipped tests (beyond stubs)
# Warn if @test_skip count changed (tracked in .gate_baseline)
# ------------------------------------------------
echo "--- Check 2: Test Health ---"

SKIP_COUNT=$(grep -rc '@test_skip' "$TEST_DIR"/test_*.jl 2>/dev/null | awk -F: '{s+=$2} END {print s}')
BASELINE_FILE="$PROJECT_ROOT/.gate_baseline"

if [ -f "$BASELINE_FILE" ]; then
    PREV_SKIP=$(grep 'skip_count' "$BASELINE_FILE" | awk -F= '{print $2}')
    if [ "$SKIP_COUNT" -gt "$PREV_SKIP" ]; then
        echo -e "  ${RED}FAIL: Skip count increased ($PREV_SKIP -> $SKIP_COUNT)${NC}"
        FAIL=$((FAIL + 1))
    elif [ "$SKIP_COUNT" -lt "$PREV_SKIP" ]; then
        echo -e "  ${GREEN}PASS: Skip count decreased ($PREV_SKIP -> $SKIP_COUNT) — tests being implemented${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "  ${GREEN}PASS: Skip count unchanged ($SKIP_COUNT)${NC}"
        PASS=$((PASS + 1))
    fi
else
    echo -e "  ${YELLOW}WARN: No baseline yet. Creating .gate_baseline${NC}"
    WARN=$((WARN + 1))
fi

# Update baseline
echo "skip_count=$SKIP_COUNT" > "$BASELINE_FILE"
echo "last_run=$(date -Iseconds)" >> "$BASELINE_FILE"

echo ""

# ------------------------------------------------
# Check 3: Run Julia tests
# ------------------------------------------------
echo "--- Check 3: Julia Tests ---"

if julia --project="$PROJECT_ROOT" -e 'include("test/runtests.jl")' 2>&1; then
    echo -e "  ${GREEN}PASS: All tests passed${NC}"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL: Tests failed${NC}"
    FAIL=$((FAIL + 1))
fi

echo ""

# ------------------------------------------------
# Summary
# ------------------------------------------------
echo "============================================"
echo "  Gate Summary"
echo "============================================"
echo -e "  ${GREEN}Passed: $PASS${NC}"
echo -e "  ${RED}Failed: $FAIL${NC}"
echo -e "  ${YELLOW}Warnings: $WARN${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}GATE: FAILED${NC}"
    exit 1
else
    echo -e "  ${GREEN}GATE: PASSED${NC}"
    exit 0
fi
