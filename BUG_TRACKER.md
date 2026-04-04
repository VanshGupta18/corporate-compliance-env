# 🐛 BUG TRACKER - Vedika's Day 3 Findings
## Task: Monitor and Document Issues

**Created**: April 4, 2026  
**QA Lead**: Vedika  
**Status**: All items tracked and assigned

---

## CRITICAL BLOCKERS - Must Fix Before Demo

### BUG #001: Missing `/tasks` Endpoint 🔴
- **Test Failing**: `test_tasks_endpoint_is_available`
- **HTTP Status**: 404 Not Found
- **Expected**: Should return available tasks (easy, medium, hard)
- **Current**: Endpoint not implemented
- **Severity**: **BLOCKER**
- **Assigned To**: Vansh
- **Reported**: April 4, 2026 - Vedika
- **Status**: 🔴 **OPEN**
- **Fix Steps**:
  1. Add route in `app/server/app.py`: `@app.get("/tasks")`
  2. Return list of available task IDs: `["easy", "medium", "hard"]`
  3. Include task metadata (max_steps, expected_steps, description)
  4. Re-run test: should pass
- **Est. Time**: 30 minutes

---

### BUG #002: WebSocket Connection Instability 🔴
- **Tests Failing**:
  - `test_client_can_take_a_step`
  - `test_easy_grader_correct_approval`
  - `test_penalty_for_max_steps_reached`
  - `test_reward_for_wrong_resolution`
- **Error**: `websockets.exceptions.ConnectionClosedOK`
- **Impact**: Multi-step episodes cannot complete
- **Current State**: WebSocket opens, then closes unexpectedly after 1-2 steps
- **Severity**: **BLOCKER**
- **Assigned To**: Vansh
- **Reported**: April 4, 2026 - Vedika
- **Status**: 🔴 **OPEN**
- **Investigation Steps**:
  1. Check `EnvClient.sync()` implementation - is it properly managing async lifecycle?
  2. Verify WebSocket keep-alive settings
  3. Check for premature connection closure in client code
  4. Review OpenEnv client wrapper in `app/client.py`
- **Est. Time**: 1-2 hours (debugging required)

---

### BUG #003: Response Format Mismatch 🔴
- **Tests Failing**:
  - `test_reset_endpoint_works_for_all_tasks` (assertion error)
  - `test_state_endpoint_is_available` (wrong structure)
  - `test_client_can_reset_environment` (assertion error)
- **Issue**: Tests expect certain field structure, endpoint returns different format
- **Severity**: **BLOCKER**
- **Assigned To**: Vansh + Vedika
- **Reported**: April 4, 2026 - Vedika
- **Status**: 🔴 **OPEN**
- **Debug Steps** (Vedika to run):
  1. Check actual response from `/reset` endpoint
  2. Compare with expected schema in Pydantic models
  3. Determine if tests are wrong OR endpoints are wrong
  4. Report findings back to Vansh
- **Est. Time**: 1 hour (debugging)

---

## MEDIUM PRIORITY - Nice to Have

### BUG #004: Grader Tests Not Fully Implemented ⚠️
- **Test Failing**: Multiple tests in `test_graders.py`
- **Issue**: Tests are still placeholders, not actual grader validation
- **Impact**: Cannot verify grading logic
- **Severity**: **MEDIUM**
- **Assigned To**: Both Vansh (for grader) + Vedika (for tests)
- **Status**: 🟡 **OPEN - LOWER PRIORITY**
- **Note**: These can be completed after #001-#003 are fixed
- **Est. Time**: 2-3 hours (for full implementation)

---

## LOW PRIORITY - Post-Demo

### BUG #005: No Docker Support ⚠️
- **Issue**: No Dockerfile present
- **Impact**: Cannot containerize application for deployment
- **Severity**: **LOW**
- **Assigned To**: Vansh
- **Status**: 🟡 **OPEN - POST-DEMO**
- **Action Required**: Create Dockerfile for release version
- **Est. Time**: 1 hour (standard setup)

---

## CLOSED ISSUES ✅

### ✅ RESOLVED: Data Validation Issues
- **Issue**: 100 claims needed validation against policy
- **Reported**: Day 2
- **Resolved**: Day 3 by Sanya
- **Status**: 🟢 **CLOSED - FROZEN**
- **Validation**: All 100 claims ground truth verified, dataset frozen

### ✅ RESOLVED: Dataset Field Mapping
- **Issue**: Claim IDs and field names uncertain
- **Reported**: Day 1
- **Resolved**: Day 2 via Sanya validation
- **Status**: 🟢 **CLOSED**
- **Result**: All fields validated, mappings confirmed correct

---

## BURN-DOWN STATUS

### Critical Path to Demo

```
Current State:  2/14 tests passing (14%)
               ↓
With Fix #001:  4/14 tests passing (29%) - `/tasks` endpoint added
               ↓
With Fix #002:  8/14 tests passing (57%) - WebSocket fixed
               ↓
With Fix #003:  10/14 tests passing (71%) - Response formats corrected
               ↓
Target State:   14/14 tests passing (100%) - All tests pass
```

### Time Estimate

- **Fix #001**: 30 mins → ~4 tests fixed
- **Fix #002**: 1-2 hours → ~4 tests fixed
- **Fix #003**: 1 hour → ~2 tests fixed
- **Re-test & Verify**: 30 mins
- **TOTAL**: 3-4 hours to full test pass

---

## TEST RESULTS SNAPSHOT

```
Day 3 Test Run (April 4, 06:00 PM)
===================================
Total: 14 tests
Passed: 2 ✅
Failed: 12 ❌
Pass Rate: 14.3%

Modules with Issues:
- test_api.py: 3 failures (3 tests, 0 passed)
- test_client.py: 2 failures (2 tests, 0 passed)
- test_graders.py: 3 failures (4 tests, 1 passed)
- test_rewards.py: 4 failures (5 tests, 1 passed)

Next Test Run: After Vansh fixes #001-#003
Expected: 10-14 tests passing
```

---

## MANUAL VALIDATION RESULTS ✅

All manual tests performed with success (see below).

### Endpoint Response Verification

✅ **Health Endpoint**
```
curl http://localhost:8000/health
Response: {"status":"healthy"}
Status: WORKING
```

✅ **Reset Endpoint (Easy)**
```
curl -X POST http://localhost:8000/reset -d '{"task_id":"easy"}'
Response: Valid observation with ticket_id, amount, description
Status: WORKING
```

✅ **Reset Endpoint (Medium)**
```
curl -X POST http://localhost:8000/reset -d '{"task_id":"medium"}'
Response: Valid observation for multi-step task
Status: WORKING
```

✅ **Reset Endpoint (Hard)**
```
curl -X POST http://localhost:8000/reset -d '{"task_id":"hard"}'
Response: Valid observation with missing_document field
Status: WORKING
```

✅ **Step Endpoint**
```
curl -X POST http://localhost:8000/step -d '{"action_type":"SearchPolicy","query":"expense"}'
Response: Returns observation with reward and done flag
Status: WORKING
```

---

## RECOMMENDATIONS FOR VANSH

### Priority 1 (Today)
1. Implement `/tasks` endpoint - **30 mins**
   - Route: `@app.get("/tasks")`
   - Return: List of task definitions from `openenv.yaml`

2. Debug WebSocket connection - **1-2 hours**
   - Check `EnvClient.sync()` method
   - Verify async lifecycle management
   - Add logging for connection events

3. Validate response formats - **1 hour**
   - Compare actual vs expected responses
   - Update tests OR fix endpoints based on findings

### Priority 2 (Tomorrow)
1. Implement full grader tests - **2-3 hours**
2. Add Docker support - **1 hour**
3. Finalize README examples - **1 hour**

---

## README.md Updates Needed

Once fixes are complete, update README with:

```markdown
## Test Results

After completing setup, run the test suite:

### Prerequisites
- Server must be running: `uv run uvicorn app.server.app:app --reload`
- Database/data must be loaded (automatic on first run)

### Running Tests
\`\`\`bash
# Run all tests
uv run pytest -v

# Run specific test module
uv run pytest -v tests/test_graders.py

# Expected output: 14 passed tests
\`\`\`

### Known Issues (Being Fixed)
- If tests fail with connection errors, ensure server is running on port 8000
```

---

## Sign-Off

**QA Lead**: Vedika  
**Date**: April 4, 2026  
**Status**: 🟠 **3 BLOCKERS IDENTIFIED AND TRACKED**

All issues have been logged, severity assigned, and owners designated. Ready for Vansh to begin fixes.

**Expected Full Pass**: April 4, 2026 - EOD (after fixes applied)
