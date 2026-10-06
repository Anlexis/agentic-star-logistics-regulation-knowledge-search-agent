# Logistics Regulation & Knowledge Search Agent

AI agent for searching logistics knowledge bases and regulations, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Logistics
> **Template ID**: LOG-C2-005

## Overview

Answers natural-language questions about logistics regulations from a curated,
seeded knowledge base — customs regulations, IATA/IMDG dangerous-goods rules, carrier
tariffs, Incoterms, and country-specific import rules. A trade-compliance officer,
customs broker, or logistics coordinator asks a question such as "What packing group
and UN number applies to lithium batteries by sea, and what documentation does IMDG
require?" and receives a source-cited answer assembled from the most relevant
knowledge-base passages, with an advisory disclaimer attached to every response.

Retrieval is deterministic and network-free: a field-weighted keyword score over the
seeded knowledge base, plus an exact regulation-code match bonus (UN numbers, HS
headings, IATA/IMDG designators) so questions naming an exact code rank the precise
rule above loose keyword overlap. Answers are assembled only from the retrieved
passages, so every statement is grounded in the knowledge base by construction; an
LLM-synthesis upgrade seam is documented in the design specification.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

