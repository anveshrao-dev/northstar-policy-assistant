# Offline integrity checks

The [GitHub Actions workflow](../../.github/workflows/offline-integrity.yml) runs a
small Python 3.12 standard-library subset without installing application packages,
downloading models, supplying secrets or calling providers. Its name and job scope
deliberately describe integrity checks, not end-to-end RAG validation.

It checks Python syntax without executing application modules, expected package
layout, runtime/optional direct dependency pins against `uv.lock`, frozen evaluation/source hashes,
negative validator regressions, and shell syntax. Test fixtures stay in ignored
`.local/tmp`. The source check is not a wheel build or a dependency resolver;
the separately pinned build backend is outside the application's lock package list.
Documentation links are reviewed separately; this job does not crawl websites.

Reproduce the job locally with an existing Python 3.12 environment:

```bash
.venv/bin/python -I -S -B scripts/check_source.py
.venv/bin/python -I -S -B scripts/validate_evaluations.py
.venv/bin/python -I -S -B -m unittest discover -s tests -p test_validate_evaluations.py -v
.venv/bin/python -I -S -B -m unittest discover -s tests -p test_source_checks.py -v
for script in scripts/*.sh; do bash -n "$script" || exit 1; done
```

`-I -S -B` isolates the commands shown from Python environment/site packages and
avoids bytecode writes. The existing evaluation-regression subprocesses use `-B`
without inheriting `-I -S`; their validator uses only the standard library and
synthetic policy copies. No `src/` path override is used. These checks do not run
Streamlit, import graph/provider libraries, open credentials or inspect accounting.

GitHub must fetch the checkout/action implementations and provision Python when
the workflow runs. That bootstrap is distinct from the offline checks themselves.
No hosted run is claimed until the workflow is separately committed and executed.
Permissions are read-only, checkout credentials are not retained, and no deployment
or pull-request-target event is configured.

## Full validation remains separate

With the existing locked development extras installed, run the network-blocked
application suite and offline lock check:

```bash
.venv/bin/python -B scripts/test_offline.py
uv lock --check --offline
```

Use the project-local cache settings in [AGENTS.md](../../AGENTS.md) before uv.
The full-suite socket guard is not a filesystem sandbox. Northstar's Chroma adapter
uses [isolated initialization](Secure-Chroma-Initialization.md) to prevent its
dependency from reading application dotenv files. Direct third-party imports
outside that boundary can still consult local configuration. Run full-stack checks
in a credential-free test environment; fake providers alone do not guarantee file
isolation. The stdlib CI job avoids these application imports entirely.

Cached-model integrations additionally require the pinned assets and
`NORTHSTAR_RUN_MODEL_TESTS=1`. The full suite covers retrieval, policy eligibility,
grounding, graph branches, durable accounting, UI and fake-provider failure paths.
Neither suite proves live provider compatibility, browser layout, current hosted
accounting or answer faithfulness. Linux full-stack installation and memory remain
separate portability checks; this small job avoids the Torch/CUDA install footprint.
