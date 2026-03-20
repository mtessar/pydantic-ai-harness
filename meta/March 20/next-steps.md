# Pydantic Harness: Next Steps

Synthesized from huddles on March 20, 2026. Context: fundraising period — the next 1-2 weeks are critical for public perception and negotiation leverage. Everything below is prioritized through that lens.

---

## 1. Capabilities Branch & Core Abstractions (Douwe + David Montague)

**Owner:** Douwe, David Montague (working in-person in Montana)

- Merge the three foundational PRs into the capabilities branch:
  - [#4640 — Add new capabilities abstraction + make agents serializable](https://github.com/pydantic/pydantic-ai/pull/4640)
  - [#4642 — Add before/after/wrap lifecycle hooks to AbstractCapability](https://github.com/pydantic/pydantic-ai/pull/4642)
  - [#4688 — Add for_run/for_run_step lifecycle hooks to AbstractToolset](https://github.com/pydantic/pydantic-ai/pull/4688)
- Once merged, the capabilities branch becomes the base for everything downstream (harness package, issue plans, etc.)
- The core abstraction: capabilities = hooks + model settings + tool sets. Validate that every harness feature can be expressed in those terms.
  - **Question:** Are there any features in frontier agent harnesses that _cannot_ be expressed as a combination of hooks, model settings, and tool sets?
- Validate capabilities with VStorm (Kasper) and Mike on Monday (March 23). If validated, aim for a blog post announcement by Tuesday (March 24).

**Related issues/references:**
- [#4303 — New "capabilities" abstraction](https://github.com/pydantic/pydantic-ai/issues/4303)
- [#4233 — Pydantic AI v2 Traits API: Research Report & Conceptual Design](https://github.com/pydantic/pydantic-ai/pull/4233)
- [#2885 — Middlewares or hooks for processing model requests/responses](https://github.com/pydantic/pydantic-ai/issues/2885)

---

## 2. Separate Harness Package (`pydantic-capabilities` or similar)

**Owner:** Douwe (architecture), David SF + Aditya (implementation pipeline)

Ship capabilities and high-level harness components in a **separate package** with independent versioning (v0 range), decoupled from pydantic-ai release cadence. Modeled after Langchain's `langchain-deep-agents` pattern.

- Depends on `pydantic-ai-slim`; imports the `AbstractCapability` class and related infrastructure
- Exposes combinable, high-level capabilities — not low-level primitives
  - Good: `AutoThink`, `CodeMode`, `FullMemorySystem`
  - Avoid: `ThinkingCapability` (too granular — users shouldn't need to "add thinking" as boilerplate)
- Capabilities should be composable: a "combined capability" bundles multiple primitives (e.g., Code Mode = code execution + REPL + tool sets + specific hooks)
- **Question:** What's the package name? `pydantic-capabilities`? `pydantic-harness`? Something else?
- **Question:** Do we need the package bootstrapped before issue generation, or can issue generation proceed independently? (Douwe says issues can be created before any Python code exists in the repo.)

---

## 3. Issue Generation — Research & Catalog All Harness Features

**Owner:** Aditya (primary), with input from Douwe

Goal: produce a comprehensive set of issues covering every feature a frontier agent harness should have, informed by research into existing frameworks.

### 3.1 Research phase
- Feed the capabilities issue ([#4303](https://github.com/pydantic/pydantic-ai/issues/4303)), hooks issue ([#2885](https://github.com/pydantic/pydantic-ai/issues/2885)), and the V2 plan PR ([#4233](https://github.com/pydantic/pydantic-ai/pull/4233)) as upstream context
- Research state-of-the-art agent harnesses/frameworks (Claude Code, Devin, OpenAI Codex, SWE-agent, Aider, etc.)
- Identify every feature/capability that should exist in a competitive harness
- This can be a local Claude session — does not need to be reproducible

### 3.2 Issue creation
- 6 core capabilities already identified: **Instructions**, **Model Settings**, **Thinking**, **Whispers**, **Tool Sets**, **History Processors**
- Additional capabilities discussed but not yet on the capabilities branch: **Channels**, **Background Tasks**
- **Question:** Is `tool_choice` a standalone capability or part of a higher-level "force tool on first call" capability?
- Each issue should include:
  - What the capability does at a high level
  - How it maps to the capabilities abstraction (hooks + model settings + tool sets)
  - Dependencies on foundation work
  - Open questions / ambiguities flagged explicitly
- Target: ~10-20 well-scoped issues covering the full harness surface area
- **Timeline:** Have issues created by end of March 20 so there's a starting point for discussion on March 21

---

## 4. Plan Generation Pipeline ("Software Factory")

**Owner:** David SF (pipeline architecture), Aditya (multi-model extension)

Build an automated pipeline that goes from issue to plan to implementation.

### 4.1 Plan generation (priority)
- For each issue, generate a detailed implementation plan
- Multi-model approach: Claude (Opus) + GPT-5.4 both generate plans, then debate/critique each other's plans
  - Identify disagreements and ambiguities
  - Consolidate into a single plan with all points addressed
- Each plan decision must be **backed by a reference** (link to code, issue, docs, or external framework)
- Ambiguous or unresolved questions should be raised as PR comments for human review
- Plans live as PRs in this repository

### 4.2 Implementation trigger
- Label-based trigger: when a "ready-to-implement" label is added to a plan PR, trigger Claude Code via GitHub Actions to implement the plan
- Before labeling: plan is in draft, open for comments and refinement
- After labeling: automated implementation kicks off

### 4.3 Architecture
- David SF's Ralph loop as the base — adapt it for this repository
- GitHub as the control plane: issues, PRs, labels, Actions
- Repo should have the GitHub Actions workflow set up **before** any Python code
- Look at [Bill Easton's agentic GitHub Actions](https://github.com/bill-easton) for inspiration (exact link TBD — David SF or Aditya to share)
- **Question:** Should the pipeline be portable to pydantic-ai eventually, or is this repo-specific?

### 4.4 Aditya's local extension
- Aditya is building a local pydantic-ai extension for AI-driven issue/plan generation
- Can start local, migrate to GitHub Actions later
- Useful for rapid iteration on prompts that produce good plans

---

## 5. Code Mode with Monty (REPL)

**Owner:** Aditya

High-visibility launch candidate — "Code Mode" powered by the Monty execution environment.

- Get the REPL working end-to-end with Monty
- PR in progress: [#4153 — CodeExecutionToolset with Code Mode support](https://github.com/pydantic/pydantic-ai/pull/4153)
- Related: [#4755 — @tool -> signature and schema -> signature for Code Mode](https://github.com/pydantic/pydantic-ai/pull/4755)
- Iterate on reliability: fix tests, make it robust enough to ship
- OK to keep execution environments stuff private/internal for now if needed — shipping a working Code Mode with Monty is the priority
- Can ship as part of the separate harness package (v0) even if the API isn't fully polished
- **Target:** Ship by Monday March 23 or Tuesday March 24 at latest (earlier = better for buzz)
- **Consideration:** Douwe will review once the REPL stuff is working; expect a review cycle

---

## 6. GenAI Prices

**Owner:** Alex (primary), Aditya (support)

- GenAI prices is in a bad state and needs attention
- Alex to prioritize fixing it in the next couple of days
- Aditya available to help Alex as needed, but Code Mode takes priority over GenAI prices
- Longer-term idea: GenAI prices could become the canonical "model metadata" package, with pydantic-ai consuming it for model profiles (context window sizes, etc.)
  - **Question:** Is this worth scoping out now, or is it a post-fundraising project?

---

## 7. Regular Milestone Work

**Owner:** David SF (primary)

Keep the regular milestone moving unless something higher-priority emerges.

- [#4053 — Streaming cancellation support](https://github.com/pydantic/pydantic-ai/pull/4053) — people have been waiting for this
- [#4090 — Tool Search Toolset](https://github.com/pydantic/pydantic-ai/pull/4090) — "still a big deal" per Douwe
- Other milestone items as appropriate
- **Question:** Should David SF shift to something more impactful? Douwe to decide based on what emerges.

---

## 8. Real-Time Voice/Audio/TTS

**Owner:** Unassigned (Douwe to plan, David SF possible implementer)

- Marcelo's PR: [#4375 — Add realtime speech-to-speech API support](https://github.com/pydantic/pydantic-ai/pull/4375)
- Also: [#4386 — Add OpenTelemetry instrumentation to realtime models](https://github.com/pydantic/pydantic-ai/pull/4386)
- Problem: the current PR bypasses the agent graph entirely — calls tools directly without going through graph state transitions
- Needs architectural thinking: tool call plumbing inside the agent graph's "call tools" nodes may need to be refactored/extracted
- This is a core feature change — cannot YOLO ship in a v0 package
- **Status:** Too large for this sprint. Douwe to create an architectural plan; David SF could pick up implementation once a plan exists.
- **Question:** Can we scope a minimal version that's shippable sooner, or is this inherently all-or-nothing?

---

## 9. Marketing & Buzz

**Owner:** Everyone

- Goal: daily blog posts next week about new launches/developments
- Even spinning existing work as "new" is worth a blog post
- High-impact launches in order of buzz potential:
  1. Capabilities announcement (if validated Monday)
  2. Code Mode with Monty
  3. Any other new feature drops (drip throughout the week)
- Aditya's earlier Reddit post was effective — more of that energy
- Monty has significant industry attention (OpenAI, Anthropic aware of it) — leverage that
- **Question:** Who is writing the blog posts? Douwe? Marketing? Need a plan for content production.

---

## 10. Process / Coordination

- Douwe + David Montague in Montana doing 16-hour days through the weekend
- Aditya in Ireland until March 27 (on leave from March 27; may need to return to India earlier for personal reasons). Can maintain overlap by keeping jet-lag schedule if needed.
- David SF has a concert March 20 evening but will work before/after; available normally from March 21
- David SF to share Ralph loop setup instructions in the team channel so Douwe, Samuel, and David Montague can also use it
- Open source PR triage is deprioritized for this week — community can wait
- Monday March 23: meeting with Kasper (VStorm) to validate capabilities
- **This transcript should be fed into Claude to generate comprehensive issues for the new repository** (action item from the call)

---

## Immediate Action Items (March 20 evening / March 21)

| Who | What | By When |
|-----|------|---------|
| Aditya | Get Code Mode REPL working with Monty, iterate on tests/reliability | March 23 |
| Aditya | Run issue generation session: research frameworks, create ~10-20 issues in this repo | March 21 |
| David SF | Extract capabilities list from capabilities branch, create overview issue | March 21 |
| David SF | Sketch out GitHub Actions-based code factory pipeline (based on Ralph loop) | March 21 |
| David SF | Share Ralph loop instructions in team channel | March 20 |
| David SF | Continue milestone work (streaming cancellation, tool search) | Ongoing |
| Douwe | Merge hooks + set_state into capabilities branch | March 21 |
| Douwe | Continue capabilities branch work with David Montague | March 21-25 |
| Douwe | Validate capabilities with VStorm/Mike on Monday | March 23 |
| Douwe | Prepare capabilities blog post | March 24 |
| Alex | Prioritize GenAI prices fixes | March 21-22 |
| Someone | Feed this transcript + context into Claude for comprehensive issue generation | March 20-21 |
