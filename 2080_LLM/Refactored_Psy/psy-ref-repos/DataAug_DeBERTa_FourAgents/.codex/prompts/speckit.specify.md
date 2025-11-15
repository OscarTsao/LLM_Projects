---
description: Create or update the feature specification from a natural language feature description.
---

## User Input

```text
# Functional Specification — Four-Agent Pipeline
_Last updated: 2025-10-17 09:15:49 UTC_

## Feature
**Name:** `four-agent-pipeline` (non‑conversational MVP)

## Objective
Given sentence‑labeled posts, produce:
1) sentence‑level **EvidenceUnits** (present/absent + confidence + provenance),
2) a calibrated **CriteriaResult** with decision and cited support,
3) **Top‑K suggestions** for which symptoms to verify next,
4) **Evaluation** metrics, calibration artifacts, and structured prediction outputs.

## Inputs
- Dataset in JSONL or CSV resolved to in‑memory objects with:
  - `post_id: str`
  - `sentences: List[{"sentence_id": str, "text": str}]`
  - `labels: List[{"sentence_id": str, "symptom": str, "status": int}]`  // 1=present, 0=explicit-absent
- **Symptoms (initial set)**: DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS.
- **Splits**: GroupKFold 80/10/10 by `post_id`, persisted to disk.

## Outputs (Files)
- **HPO:** `outputs/hpo/{study_name}/best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`
- **Training:** `outputs/training/{run_id}/model.ckpt`, `config.yaml`, `val_metrics.json`
- **Evaluation:** `outputs/evaluation/{run_id}/predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`
- **Calibration:** `artifacts/calibration.json` (temperature + per‑symptom thresholds)

## Output Schemas
- **predictions.jsonl** (per (post, sentence, symptom)):
  ```json
  {"post_id":"P1","sentence_id":"S3","symptom":"SLEEP_ISSUES","assertion":"present","score":0.82,"gold":1}
  ```
- **criteria.jsonl** (per post):
  ```json
  {"post_id":"P1","p_dx":0.67,"decision":"likely",
    "supporting":{"DEPRESSED_MOOD":["EU_12"],"ANHEDONIA":["EU_19"]},
    "conflicts":["APPETITE_CHANGE"],"missing":["SLEEP_ISSUES"]}
  ```

## Functional Requirements
### Evidence Agent
- Pairwise cross‑encoder over `(sentence, symptom)` → `present/absent` probability.
- Writes `predictions.jsonl` with required schema.
- Hydra‑driven training/inference; Optuna HPO on model family and hyperparameters.

### Criteria Agent
- Aggregates EvidenceUnits into features (max score per symptom, any_present/absent, conflicts).
- JSON rule counts + small classifier → provisional probability `p_dx`.
- Temperature scaling and per‑symptom thresholds from `artifacts/calibration.json` (if present).
- Writes `criteria.jsonl` with supporting EvidenceUnit ids.

### Suggestion Agent
- Value‑of‑evidence (|Δp|) counterfactuals for uncertain symptoms (configurable band).
- Emits Top‑K suggestions per post with reason strings.

### Evaluation Agent
- Evidence metrics: per‑symptom P/R/F1 (present), negation precision.
- Criteria metrics: AUROC/F1; ECE; contradiction rate; faithfulness %.
- Fits calibration (temperature + thresholds) on dev; writes `artifacts/calibration.json`.
- Writes `val_metrics.json` and `test_metrics.json` to the appropriate output directories.

## Non‑Functional Requirements
- Reproducibility: seeds, deterministic splits persisted.
- Configurability: Hydra across all components.
- Observability: MLflow logs params, metrics, and artifacts to `./mlruns`.

## Success Criteria
- Evidence macro‑F1 (present) improved ≥10 points over rules baseline.
- Negation precision ≥ 0.90.
- Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after calibration.
- 100% "likely" decisions are grounded in ≥1 present EvidenceUnit.

## Risks & Mitigations
- **Data scarcity:** start with public sample; later expand via gated dataset access.
- **Calibration drift:** run Evaluation after any Evidence retrain; refresh `artifacts/calibration.json`.
- **Label imbalance:** focal loss / class weights; per‑symptom thresholds.
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

The text the user typed after `/speckit.specify` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `$ARGUMENTS` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

1. **Generate a concise short name** (2-4 words) for the branch:
   - Analyze the feature description and extract the most meaningful keywords
   - Create a 2-4 word short name that captures the essence of the feature
   - Use action-noun format when possible (e.g., "add-user-auth", "fix-payment-bug")
   - Preserve technical terms and acronyms (OAuth2, API, JWT, etc.)
   - Keep it concise but descriptive enough to understand the feature at a glance
   - Examples:
     - "I want to add user authentication" → "user-auth"
     - "Implement OAuth2 integration for the API" → "oauth2-api-integration"
     - "Create a dashboard for analytics" → "analytics-dashboard"
     - "Fix payment processing timeout bug" → "fix-payment-timeout"

2. Run the script `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS"` from repo root **with the short-name argument** and parse its JSON output for BRANCH_NAME and SPEC_FILE. All file paths must be absolute.

   **IMPORTANT**:

   - Append the short-name argument to the `.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS"` command with the 2-4 word short name you created in step 1
   - Bash: `--short-name "your-generated-short-name"`
   - PowerShell: `-ShortName "your-generated-short-name"`
   - For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot")
   - You must only ever run this script once
   - The JSON is provided in the terminal as output - always refer to it to get the actual content you're looking for

3. Load `.specify/templates/spec-template.md` to understand required sections.

4. Follow this execution flow:

    1. Parse user description from Input
       If empty: ERROR "No feature description provided"
    2. Extract key concepts from description
       Identify: actors, actions, data, constraints
    3. For unclear aspects:
       - Make informed guesses based on context and industry standards
       - Only mark with [NEEDS CLARIFICATION: specific question] if:
         - The choice significantly impacts feature scope or user experience
         - Multiple reasonable interpretations exist with different implications
         - No reasonable default exists
       - **LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total**
       - Prioritize clarifications by impact: scope > security/privacy > user experience > technical details
    4. Fill User Scenarios & Testing section
       If no clear user flow: ERROR "Cannot determine user scenarios"
    5. Generate Functional Requirements
       Each requirement must be testable
       Use reasonable defaults for unspecified details (document assumptions in Assumptions section)
    6. Define Success Criteria
       Create measurable, technology-agnostic outcomes
       Include both quantitative metrics (time, performance, volume) and qualitative measures (user satisfaction, task completion)
       Each criterion must be verifiable without implementation details
    7. Identify Key Entities (if data involved)
    8. Return: SUCCESS (spec ready for planning)

5. Write the specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the feature description (arguments) while preserving section order and headings.

6. **Specification Quality Validation**: After writing the initial spec, validate it against quality criteria:

   a. **Create Spec Quality Checklist**: Generate a checklist file at `FEATURE_DIR/checklists/requirements.md` using the checklist template structure with these validation items:
   
      ```markdown
      # Specification Quality Checklist: [FEATURE NAME]
      
      **Purpose**: Validate specification completeness and quality before proceeding to planning
      **Created**: [DATE]
      **Feature**: [Link to spec.md]
      
      ## Content Quality
      
      - [ ] No implementation details (languages, frameworks, APIs)
      - [ ] Focused on user value and business needs
      - [ ] Written for non-technical stakeholders
      - [ ] All mandatory sections completed
      
      ## Requirement Completeness
      
      - [ ] No [NEEDS CLARIFICATION] markers remain
      - [ ] Requirements are testable and unambiguous
      - [ ] Success criteria are measurable
      - [ ] Success criteria are technology-agnostic (no implementation details)
      - [ ] All acceptance scenarios are defined
      - [ ] Edge cases are identified
      - [ ] Scope is clearly bounded
      - [ ] Dependencies and assumptions identified
      
      ## Feature Readiness
      
      - [ ] All functional requirements have clear acceptance criteria
      - [ ] User scenarios cover primary flows
      - [ ] Feature meets measurable outcomes defined in Success Criteria
      - [ ] No implementation details leak into specification
      
      ## Notes
      
      - Items marked incomplete require spec updates before `/speckit.clarify` or `/speckit.plan`
      ```
   
   b. **Run Validation Check**: Review the spec against each checklist item:
      - For each item, determine if it passes or fails
      - Document specific issues found (quote relevant spec sections)
   
   c. **Handle Validation Results**:
      
      - **If all items pass**: Mark checklist complete and proceed to step 6
      
      - **If items fail (excluding [NEEDS CLARIFICATION])**:
        1. List the failing items and specific issues
        2. Update the spec to address each issue
        3. Re-run validation until all items pass (max 3 iterations)
        4. If still failing after 3 iterations, document remaining issues in checklist notes and warn user
      
      - **If [NEEDS CLARIFICATION] markers remain**:
        1. Extract all [NEEDS CLARIFICATION: ...] markers from the spec
        2. **LIMIT CHECK**: If more than 3 markers exist, keep only the 3 most critical (by scope/security/UX impact) and make informed guesses for the rest
        3. For each clarification needed (max 3), present options to user in this format:
        
           ```markdown
           ## Question [N]: [Topic]
           
           **Context**: [Quote relevant spec section]
           
           **What we need to know**: [Specific question from NEEDS CLARIFICATION marker]
           
           **Suggested Answers**:
           
           | Option | Answer | Implications |
           |--------|--------|--------------|
           | A      | [First suggested answer] | [What this means for the feature] |
           | B      | [Second suggested answer] | [What this means for the feature] |
           | C      | [Third suggested answer] | [What this means for the feature] |
           | Custom | Provide your own answer | [Explain how to provide custom input] |
           
           **Your choice**: _[Wait for user response]_
           ```
        
        4. **CRITICAL - Table Formatting**: Ensure markdown tables are properly formatted:
           - Use consistent spacing with pipes aligned
           - Each cell should have spaces around content: `| Content |` not `|Content|`
           - Header separator must have at least 3 dashes: `|--------|`
           - Test that the table renders correctly in markdown preview
        5. Number questions sequentially (Q1, Q2, Q3 - max 3 total)
        6. Present all questions together before waiting for responses
        7. Wait for user to respond with their choices for all questions (e.g., "Q1: A, Q2: Custom - [details], Q3: B")
        8. Update the spec by replacing each [NEEDS CLARIFICATION] marker with the user's selected or provided answer
        9. Re-run validation after all clarifications are resolved
   
   d. **Update Checklist**: After each validation iteration, update the checklist file with current pass/fail status

7. Report completion with branch name, spec file path, checklist results, and readiness for the next phase (`/speckit.clarify` or `/speckit.plan`).

**NOTE:** The script creates and checks out the new branch and initializes the spec file before writing.

## General Guidelines

## Quick Guidelines

- Focus on **WHAT** users need and **WHY**.
- Avoid HOW to implement (no tech stack, APIs, code structure).
- Written for business stakeholders, not developers.
- DO NOT create any checklists that are embedded in the spec. That will be a separate command.

### Section Requirements

- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation

When creating this spec from a user prompt:

1. **Make informed guesses**: Use context, industry standards, and common patterns to fill gaps
2. **Document assumptions**: Record reasonable defaults in the Assumptions section
3. **Limit clarifications**: Maximum 3 [NEEDS CLARIFICATION] markers - use only for critical decisions that:
   - Significantly impact feature scope or user experience
   - Have multiple reasonable interpretations with different implications
   - Lack any reasonable default
4. **Prioritize clarifications**: scope > security/privacy > user experience > technical details
5. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
6. **Common areas needing clarification** (only if no reasonable default exists):
   - Feature scope and boundaries (include/exclude specific use cases)
   - User types and permissions (if multiple conflicting interpretations possible)
   - Security/compliance requirements (when legally/financially significant)
   
**Examples of reasonable defaults** (don't ask about these):

- Data retention: Industry-standard practices for the domain
- Performance targets: Standard web/mobile app expectations unless specified
- Error handling: User-friendly messages with appropriate fallbacks
- Authentication method: Standard session-based or OAuth2 for web apps
- Integration patterns: RESTful APIs unless specified otherwise

### Success Criteria Guidelines

Success criteria must be:

1. **Measurable**: Include specific metrics (time, percentage, count, rate)
2. **Technology-agnostic**: No mention of frameworks, languages, databases, or tools
3. **User-focused**: Describe outcomes from user/business perspective, not system internals
4. **Verifiable**: Can be tested/validated without knowing implementation details

**Good examples**:

- "Users can complete checkout in under 3 minutes"
- "System supports 10,000 concurrent users"
- "95% of searches return results in under 1 second"
- "Task completion rate improves by 40%"

**Bad examples** (implementation-focused):

- "API response time is under 200ms" (too technical, use "Users see results instantly")
- "Database can handle 1000 TPS" (implementation detail, use user-facing metric)
- "React components render efficiently" (framework-specific)
- "Redis cache hit rate above 80%" (technology-specific)
