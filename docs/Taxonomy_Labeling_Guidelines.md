# BudgetMind — Task Classification Taxonomy & Labeling Guidelines

**Version:** 1.0 · Sprint 1 
**Team:** GRNEE · Author: Rawda Mohamed

- [1. Purpose and Scope](#1-purpose-and-scope)
- [2. Final Taxonomy](#2-final-taxonomy)
  - [2.1 TYPE](#21-type)
  - [2.2 COMPLEXITY](#22-complexity)
  - [2.3 DOMAIN](#23-domain)
- [3. Labeling Procedure](#3-labeling-procedure)
- [4. Tie-Breaking Rules](#4-tie-breaking-rules) 
- [5. Gold Examples](#5-gold-examples)
- [6. Quick Reference](#6-quick-reference)

---

## 1. Purpose and Scope

BudgetMind's Task Classifier reads each incoming task and produces three labels — **TYPE**, **COMPLEXITY**, and **DOMAIN** — which are consumed by the Routing Policy Learner when it selects a model.

This document defines those three labels and the rules for applying them.

*Out of scope* (these belong to the Routing Layer, not the classifier):

- Which model gets picked 
- Cost, latency, and budget constraints
- User preferences, preferred or blocked models, security eligibility 
- Fallback behavior

Every task must receive **exactly one label per dimension**.

---

## 2. Final Taxonomy

### 2.1 TYPE

TYPE describes the cognitive operation the model is being asked to perform.

| # | Label | Definition | Examples |
|---|-------|-----------|----------|
| 1 | **Factual Q&A** | A single, self-contained request for a specific piece of information. | "What is the capital of Australia?" · "What does MCP stand for?" |
| 2 | **Translation** | Converting text from one language to another. | "Translate this paragraph into French." | 
| 3 | **Summarization** | Condensing a supplied text into a shorter version. | "Summarize this article in 3 bullets." · "TL;DR of this report." |
| 4 | **Coding & Debugging** | Writing, fixing, refactoring, or reviewing source code, queries, or scripts. | "Fix this Python function." · "Write a SQL query to find duplicates." |
| 5 | **Mathematical Reasoning** | Solving a problem whose core mechanism is arithmetic, algebra, or quantitative computation. | "Solve for x: 3x + 7 = 22." · GSM8K-style word problems. | 
| 6 | **Analytical / Logical Reasoning** | Applying logic, judgment, or multi-step inference without a numeric-computation core. | "Which strategy carries the most regulatory risk, and why?" · MMLU-style questions. |
| 7 | **Architecture & System Design** | Planning the high-level structure of a system — components, trade-offs, non-functional requirements — where the deliverable is a design, not code. | "Design a scalable backend for a ride-sharing app." · "Monolith vs. microservices for this use case?" |
| 8 | **Creative & Open-Ended Generation** | Producing original content where there is no single correct answer and style or originality is the point. | "Write a launch email in an upbeat tone." · "Brainstorm 10 app names." |


### 2.2 COMPLEXITY 

COMPLEXITY estimates how hard the task is for a model to handle well. **It does not depend on prompt length.** A one-line "design a fault-tolerant payment system" is High; a five-paragraph "translate the following" is Low.

Signals to weigh:

- **Reasoning depth** — how many inferential steps are needed
- **Specialized knowledge** — is common knowledge enough, or is expert domain knowledge required?  
- **Ambiguity** — is the ask well-specified, or must the solver interpret it?
- **Coordinated steps and constraints** — how many things must be juggled at once?
- **Consequence of error** — is a wrong answer easy to catch and cheap to fix, or costly and hard to detect?

| Label | Definition |
|-------|------------|
| **Low** | Single-step or near-single-step. No specialized expertise required. Little to no ambiguity. Mistakes are easy to spot and low-stakes. |
| **Medium** | Several coordinated steps, or moderate domain knowledge required. Task is well-specified but not trivial. Mistakes are noticeable but not high-stakes. Specialized single-step lookups sit here at minimum. | 
| **High** | Deep or specialized expertise, unresolved ambiguity or real trade-offs, or high error cost. Multi-step planning is typical. |

Any single strong High signal (deep expertise, unresolved trade-offs, high error cost) makes the task High even if other signals are moderate.

### 2.3 DOMAIN

DOMAIN describes the subject-matter area the task belongs to. 

For Sprint 1, four broad domains are used to provide consistent, practical classification while avoiding unnecessary fragmentation. The taxonomy can be extended later if analysis of the real dataset reveals recurring specialized areas that materially affect routing decisions.

| # | Label | Definition | Examples |
|---|-------|-----------|----------|
| 1 | **Software Engineering & Technology** | Subject matter is code, systems, databases, DevOps, APIs, or IT infrastructure. | "Design a caching strategy." · "Fix this null pointer exception." · "REST vs. gRPC?" |
| 2 | **Mathematics & Quantitative** | Subject matter is math, statistics, or numeric computation, and the deliverable is a number or derivation (not code). | "Compute standard deviation by hand." · "Solve for x." |
| 3 | **Language & Communication** | Task is fundamentally about language itself — translation, grammar, tone, style — with no technical subject. | "Translate to French." · "Rewrite in a formal tone." |
| 4 | **General Knowledge / Cross-Domain** | Everyday knowledge, common-sense reasoning, or subject matter that doesn't fit the three above; also the default for genuinely mixed multi-domain tasks. | "Capital of Australia?" · "Which historical event happened first?" · "Help me plan a trip." |

---

## 3. Labeling Procedure

For each task, follow this order:

1. **Read the task and identify the primary deliverable** — what does the requester actually want back?
2. **Assign TYPE** using the decision order in the Quick Reference (Section 6).
3. **Assign COMPLEXITY** by weighing the signals in 2.2.  
4. **Assign DOMAIN** using the decision order in the Quick Reference.
5. **Apply the tie-break rules in Section 4** if two labels genuinely seem to fit.
6. **Add a note** only when a case isn't cleanly resolved by the tie-break rules. Notes are for team review, not a substitute for picking a label.

Every row ends with exactly one TYPE, one COMPLEXITY, one DOMAIN. 

---

## 4. Tie-Breaking Rules

| Situation | Rule |
|-----------|------|
| Coding + reasoning ("figure out why this is slow and rewrite it") | If a code artifact is the primary deliverable → **Coding & Debugging**. If only an explanation is requested → **Analytical/Logical Reasoning**. |
| Architecture + coding ("design and implement X") | If runnable code is the primary deliverable → **Coding & Debugging**. If the deliverable is a design/plan → **Architecture & System Design**. |
| Analysis + summarization ("summarize this and tell me if the conclusion holds") | Label by the primary requested output. If the summary itself is the point → **Summarization**. If the judgment is the point → **Analytical/Logical Reasoning**. | 
| Translation + explanation ("translate this and explain what it means") | If translation is the primary deliverable → **Translation**. If explanation dominates (e.g., one phrase, mostly discussion of nuance) → **Analytical/Logical Reasoning** or **Factual Q&A**. |
| Multiple requested outputs ("translate, summarize, and give your opinion") | Label by the primary requested output. If they're roughly equal, pick the type that determines the main capability needed. |
| Multi-domain task | Pick the domain that governs the most specialized part. If genuinely balanced → **General Knowledge / Cross-Domain**. |
| Specialized single-step lookup ("standard dosage of X") | TYPE = **Factual Q&A**, COMPLEXITY = **Medium** minimum (never Low for specialized knowledge). |
| Long-but-easy or short-but-hard | Ignore length. Apply the complexity signals in 2.2. | 
| Underspecified task ("help me with my project") | TYPE = **Analytical/Logical Reasoning** (the solver must interpret the ask), COMPLEXITY = **Medium** minimum, DOMAIN = **General Knowledge / Cross-Domain** unless context specifies otherwise. |
| Task carries prior conversation turns | Ignore that it's multi-turn. Label by what the *latest turn* is actually asking. |

---

## 5. Gold Examples 

The following 20 gold examples were selected to illustrate important classification boundaries and common edge cases. They serve as reference cases for maintaining consistent annotation decisions, especially when a task could reasonably fall under more than one label.

| # | Task | TYPE | COMPLEXITY | DOMAIN | Why |
|---|------|------|------------|--------|-----|
| 1 | Translate this paragraph about our return policy into Spanish. | Translation | Low | Language & Communication | Straight conversion, no specialized knowledge. |
| 2 | Design a scalable architecture for a ride-sharing backend, including surge pricing and driver-matching at scale. | Architecture & System Design | High | Software Engineering & Technology | Multiple trade-offs, deep expertise, high error cost. |
| 3 | Fix this Python function — it throws an IndexError on empty lists. | Coding & Debugging | Low | Software Engineering & Technology | Single well-specified bug, one clear fix. |
| 4 | If a train travels 60 mph for 2.5 hours, how far does it go? | Mathematical Reasoning | Low | Mathematics & Quantitative | Single-step arithmetic. |
| 5 | Given these three go-to-market strategies, which has the most EU regulatory risk, and why? | Analytical/Logical Reasoning | High | General Knowledge / Cross-Domain | Multi-step judgment, no numeric core, mixed subject matter. | 
| 6 | Summarize this 10-page quarterly earnings report into 5 bullets. | Summarization | Medium | General Knowledge / Cross-Domain | Condensation across many pages; no specialized financial judgment requested. |
| 7 | What is the capital of Australia? | Factual Q&A | Low | General Knowledge / Cross-Domain | Common-knowledge lookup. | 
| 8 | [Chat history: trip planning] User: Can we make day 2 cheaper without cutting Kyoto? | Analytical/Logical Reasoning | Medium | General Knowledge / Cross-Domain | Revising a plan under a constraint. Prior turns are context, not a TYPE. |
| 9 | Write a SQL query to find customers with more than 3 orders in the last 30 days. | Coding & Debugging | Medium | Software Engineering & Technology | Join + aggregation + date filter to coordinate correctly. |
| 10 | Write a short, upbeat product launch email for a new fitness tracker. | Creative & Open-Ended Generation | Low | Language & Communication | Stylistic, no single right answer. |
| 11 | Explain what this code does and suggest a more efficient algorithm — don't rewrite it. | Analytical/Logical Reasoning | High | Software Engineering & Technology | No code deliverable requested; requires deep algorithmic reasoning. |
| 12 | Solve for x: 3x + 7 = 22. | Mathematical Reasoning | Low | Mathematics & Quantitative | Single-step algebra. |
| 13 | Which of these four architectural patterns best fits a 3-person startup building an MVP, and why? | Architecture & System Design | Medium | Software Engineering & Technology | Constrained trade-off comparison; more than trivial, short of High. |
| 14 | Translate this German error message to English and tell me what's causing it. | Translation | Medium | Language & Communication | Translation is the primary deliverable; the diagnosis bumps complexity but not TYPE. | 
| 15 | What's the standard first-line treatment threshold for stage 2 hypertension? | Factual Q&A | Medium | General Knowledge / Cross-Domain | Single-step lookup but specialized knowledge — never Low. |
| 16 | Write a Python script to compute the standard deviation of this list. | Coding & Debugging | Low | Software Engineering & Technology | Deliverable is code using a standard computation. |
| 17 | Derive the formula for standard deviation from first principles and compute it by hand. | Mathematical Reasoning | Medium | Mathematics & Quantitative | Derivation-plus-computation; no code. |
| 18 | Brainstorm 10 possible names for a new budgeting app. | Creative & Open-Ended Generation | Low | General Knowledge / Cross-Domain | Open-ended, low-risk. |
| 19 | Given current load and cost history, propose a caching strategy keeping p99 under 200 ms within a $500/month budget. | Architecture & System Design | High | Software Engineering & Technology | Hard constraints, real trade-offs, high error cost. |
| 20 | Help me with my presentation. | Analytical/Logical Reasoning | Medium | General Knowledge / Cross-Domain | Underspecified — the solver must interpret the ask. |

---

## 6. Quick Reference

This section is designed to help Member 3 to help complete the labeling process consistently. Keep it open while labeling and refer to it whenever needed.


**TYPE**

1. Converts language? → **Translation**
2. Shortens a supplied text? → **Summarization**  
3. Produces or fixes code? → **Coding & Debugging**
4. Core mechanism is arithmetic or algebra? → **Mathematical Reasoning**
5. Designs a system without a code deliverable? → **Architecture & System Design**
6. No single right answer, style or originality is the point? → **Creative & Open-Ended Generation**
7. Multi-step logic or judgment, no numeric core? → **Analytical/Logical Reasoning**
8. Otherwise (single self-contained info request) → **Factual Q&A**

**COMPLEXITY:**

- Single step, common knowledge, no ambiguity → **Low**
- Single step but specialized knowledge → **Medium** (never Low)
- Multiple coordinated but well-specified steps → **Medium** 
- Deep expertise, real trade-offs, or high error cost → **High**
- Length is not a signal.

**DOMAIN**

1. About code, systems, or IT? → **Software Engineering & Technology**
2. Numeric deliverable, not code? → **Mathematics & Quantitative**  
3. Fundamentally about language or style with no technical subject? → **Language & Communication**
4. Otherwise, or genuinely mixed → **General Knowledge / Cross-Domain** 

**Top tie-breaks:**

- Combined ask → label by the primary requested output.
- Multi-domain, no clear winner → **General Knowledge / Cross-Domain**.
- Underspecified → **Analytical/Logical Reasoning**, Medium minimum, Cross-Domain.
- Specialized single-step lookup → **Factual Q&A**, Medium (never Low).
- Long-but-easy or short-but-hard → ignore length.
- Prior chat turns → label by the latest turn's ask.

---
