# CLAUDE.md

# Critical Rules - NO EXCEPTIONS

### Never Guess or Assume
- If you don't know something, **ask** or **search** for the answer
- Don't assume file locations, API formats, or implementation details
- Don't proceed with uncertain information - verify first

### No Mocks or Fake Data
- Use REAL DATA
- No placeholder data, mock responses, or synthetic test data
- If an API call fails, **fix it** - don't fake the response

### No Shortcuts
- No incomplete implementations marked as "TODO"
- No skipping error handling or edge cases
- No "it works on my machine" - ensure it works properly
- Test everything before claiming completion

### No Fallbacks or Graceful Degradation
- If something fails, **stop and fix it**
- Don't return partial results with a disclaimer
- Don't paper over errors with default values
- Be honest about failures - never pretend something works

## Project Files

### `.claude/plan.md`
The single source of truth for project status and task tracking.
- Contains the implementation plan and current todo list
- Mark completed tasks with `[x]`
- Mark in-progress tasks with `← IN PROGRESS`
- If a plan is complete, delete it. If a plan is being replaced, delete it and replace with new one.
- Note: the plan im referring to is the one you make with ExitPlanMode tool (your built-in plan mode that the user activates). It should NOT be some arbitrary plan that was not crafted with the ExitPlanMode tool.

### Key Documentation
- `projectspec/project.pdf` - Project proposal
- This file - Core instructions and standards

## Workflow

1. **Before coding:** Read relevant files, understand context
2. **Plan first:** Think through implementation before writing code
3. **Implement:** Write production-quality code with error handling
4. **Verify:** Test that it actually works with real data

## Quality Standards

- Production-ready code only
- Real API integrations (if any)
- Comprehensive error handling
- Honest status reporting
- Full test coverage

## When You Don't Know

If you're uncertain about:
- API endpoints or formats → Read API docs or ask
- File locations → Use `view` tool to explore
- Implementation details → Ask for clarification
- Whether something works → Test it and verify

**Never proceed with guesswork. Always verify.**


# MCP Server Reference

## Context7 MCP Server

**Purpose:** Provides up-to-date, version-specific code documentation for AI coding assistants.

**When to call:**
- Need current documentation for libraries/frameworks
- Want to avoid outdated or hallucinated code examples
- Working with version-specific APIs or syntax

**Available tools:**
- `resolve-library-id` - Maps library names to Context7-compatible IDs
- `get-library-docs` - Retrieves current documentation for specified library/topic

---

## Chrome DevTools MCP Server

**Purpose:** Controls and inspects live Chrome browser via DevTools API for AI agents.

**When to call:**
- Browser automation tasks
- Advanced debugging (network requests, console logs)
- Performance analysis and trace collection
- Screenshot capture
- DOM inspection or manipulation

**Capabilities:**
- Performance traces and analysis
- Network request inspection
- Console log retrieval
- Screenshot capture
- Puppeteer-based reliable automation
- use .gitignore