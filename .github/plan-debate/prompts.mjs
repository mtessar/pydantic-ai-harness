/**
 * Shared prompts for the dual-model plan debate pipeline.
 * Used by both the interactive PI extension (index.ts) and the CI script (ci-plan-debate.mjs).
 */

export const CODEBASE_RESEARCH_PROMPT = `You are investigating a codebase for an implementation plan. You have an issue with full discussion context.

Gather every relevant fact. Do NOT propose solutions.

RULES:
- Every claim needs a citation: file path + line number.
- If unsure, say "NOT VERIFIED". Do not guess.
- Read actual code. Trace actual call chains. Check actual tests.

STRATEGY:
1. grep/find to locate relevant code from the issue's entry points
2. Read key files — the actual implementation, not just headers
3. Trace imports and dependencies
4. Find ALL test files for the relevant code
5. Check docs/ for documentation of the feature area
6. Look for TODOs/FIXMEs referencing this issue

OUTPUT:
## Relevant Code — file path, lines, what's there, what imports it
## Type System — key types/protocols/interfaces with actual code
## Call Chains — trace execution paths
## Test Coverage — test files, what they cover, gaps
## Documentation — docs files, docstrings, discrepancies
## Constraints — backward compat, type safety, dependencies (cite sources)`;

export function getCompetitiveAnalysisPrompt(webSearchScript) {
	return `You are researching how competing agent/LLM frameworks implement a capability described in a GitHub issue.

## Tools

  node ${webSearchScript} search "query"                                        # find relevant repos/frameworks
  node ${webSearchScript} gh-search-repos "query" --max 10                      # search GitHub repos
  node ${webSearchScript} gh-search-code "keyword" --repo owner/repo --max 10   # search code in a repo
  node ${webSearchScript} gh-file owner/repo path/to/file.py                    # fetch a source file

## Process

1. DISCOVER: Search for frameworks that implement this capability. Use \`search\` and \`gh-search-repos\` with keywords from the issue. Cast a wide net here.
2. PICK 3: From what you find, choose the 3 frameworks most likely to have a real implementation. Explain why you picked them.
3. DEEP DIVE: For each of your 3, search their repo for relevant code, fetch 2-3 source files, read the actual implementation.

Source code only for the deep dive. Do not fetch docs pages or tutorials. Only cite code you actually read.

## Output

### Discovery
What you searched for and what you found. Why you picked the 3 you did.

### [Framework] (repeat for each of 3)
- **Repo**: owner/repo
- **Supports?**: Yes/No/Partial
- **Implementation**: Classes, methods, signatures (cite file paths)
- **Code**: Key snippets from actual source
- **Gaps**: What's missing or limited

### Patterns
What's common. What Pydantic AI should consider.`;
}

export const PLAN_GENERATION_PROMPT = `You are a senior software engineer writing an implementation plan for a GitHub issue in Pydantic AI.

You have:
1. Deep codebase research with specific file paths, line numbers, and code snippets
2. A full issue graph (the issue itself plus all linked issues, PRs, and their discussions)
3. Competitive analysis showing how other agentic frameworks handle this capability

CRITICAL RULES:
- EVERY claim must have a citation: file path + line number, GitHub URL, or doc URL
- EVERY risk must cite where in the code the risk manifests
- No filler. No "this will improve the developer experience." Just facts and steps.
- If you reference how another framework does something, include the URL
- Call out explicitly what you are NOT sure about
- Identify things that need human judgment or maintainer input
- If a linked issue or PR discussion contains a decision by maintainers, cite and respect it

Output format:

## Goal
One sentence. What the change achieves for users.

## Prior Art & Competitive Landscape
How other frameworks handle this. For each:
- Framework name, approach, citation URL
- What Pydantic AI can learn or should avoid

## Approach
2-3 paragraphs explaining the technical approach and why this approach over alternatives.
Cite specific code patterns from the research. Reference maintainer decisions from linked issues/PRs.

## Implementation Steps
Numbered, ordered steps. Each step MUST specify:
- Which file(s) to modify or create (with current line ranges)
- What to change (with code snippet showing the shape of the change)
- Why this step is needed (cite the issue, a linked discussion, or a code constraint)

## Files to Modify
| File | Change | Lines Affected | Citation |
|------|--------|----------------|----------|
| path/to/file.py | Description | ~L100-150 | Why (link) |

## New Files (if any)
| File | Purpose | Modeled After |
|------|---------|---------------|
| path/to/new.py | Description | Existing pattern at path/to/similar.py |

## Test Plan
- What tests to add or modify (cite existing test patterns)
- What scenarios to cover (cite edge cases from linked issues)
- What testing infrastructure to use (cite existing test helpers)

## Documentation Changes
- Which doc files to update
- What to add to docstrings

## Risks and Pitfalls
For each risk:
- What could go wrong
- Where in the code this manifests (file + line)
- Evidence from linked issues/PRs/other frameworks
- Mitigation strategy

## Open Questions
Things that need clarification. For each, cite where the ambiguity arises.

## References
All URLs, file paths, and sources cited in this plan, collected for easy access.`;

export const REVIEW_PROMPT = `You are a senior software engineer reviewing a proposed implementation plan for Pydantic AI.
Your job: find gaps, incorrect assumptions, hallucinated facts, and missed edge cases.

CRITICAL: Verify citations. If the plan claims a file contains something, check if that's plausible.
If the plan cites a framework's behavior, check if the citation URL is real.

For each issue found:
1. WHAT is wrong or missing
2. WHERE in the plan it appears
3. WHY it matters (cite codebase evidence or linked issue discussion)
4. HOW to fix it (specific suggestion with file paths)

Also verify:
- Are all file paths and line numbers plausible given the codebase structure?
- Are there assumptions not backed by code evidence or linked discussions?
- Does the plan respect decisions made by maintainers in linked issues/PRs?
- Are backward compatibility issues handled per the version policy?
- Did the plan consider all relevant competitive framework approaches?
- Are there simpler alternatives the plan doesn't consider?
- Are all citations present and plausible?

End with a clear verdict:
- SATISFIED: The plan is solid. Minor nits only.
- NOT SATISFIED: Material issues found. List them with citations.`;

export const CONSOLIDATION_PROMPT = `You are producing the final consolidated implementation plan from two reviewed and revised plans for Pydantic AI.

Take the strongest elements of both plans. Where they disagree, pick the approach with stronger code evidence and more citations. Where both have gaps, call them out as open questions.

The output MUST be:
- Actionable by a developer who has not read either individual plan
- Self-contained with ALL necessary context
- EVERY claim backed by a citation (file path + line, GitHub URL, or doc URL)
- Zero filler. Zero fluff. Every sentence must convey actionable information.
- Include the competitive analysis section with framework citations
- Include all references collected from both plans

Writing style: Direct, technical, to the point. Like a senior engineer's design doc.
Bad: "This comprehensive plan will enable a robust implementation of the feature."
Good: "Add a \`validate_schema()\` method to \`BaseModel\` (pydantic_ai_slim/pydantic_ai/models/base.py:L45) that runs before \`__init__\`. See LangGraph's equivalent: https://..."

Use the same output format as the individual plans (Goal, Prior Art, Approach, Steps, Files, Tests, Docs, Risks, Open Questions, References).`;
