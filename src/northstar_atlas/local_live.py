"""Explicit Mac-only controlled-live bootstrap; never initializes an authority."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

from . import live_access
from .live_budget import BudgetError, PersistentBudget, LINEAR, GRAPH, configured_budget
from .llm import MODEL_ID
from .project import contained_path, validate_project_root


def server_options(port: int) -> dict:
    return dict(server_address='127.0.0.1', server_port=port,
        browser_serverAddress='127.0.0.1', browser_serverPort=port,
        server_enableCORS=True, server_enableXsrfProtection=True,
        server_enableStaticServing=False, server_baseUrlPath='',
        server_headless=True, server_showEmailPrompt=False, server_fileWatcherType='none',
        browser_gatherUsageStats=False, client_showErrorDetails='none')


def check_budget(root, *, enable_graph, backend):
    """Offline existing-authority check. No private-preview probe or provider IO."""
    budget = configured_budget(root)
    from .one_shot import one_shot_selected, configured_plan, inspect_invocation, backend as approved_backend
    if one_shot_selected():
        status = inspect_invocation(budget, configured_plan(), require_browser=False)
        if enable_graph or backend != approved_backend(status) or not status.available:
            raise BudgetError()
    elif os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED', '0') != '0':
        from .demo_batch import activation, remaining
        if enable_graph or backend != 'pinecone' or remaining(budget, activation()) <= 0:
            raise BudgetError()
    else:
        budget.check_ready(profile=GRAPH if enable_graph else LINEAR)
    return budget


def run_question(budget, spec, *, opener, limits, diagnostic_approval=None):
    """Reserve → open → ask once → close → settle → record → return for display."""
    from .qa import QASuccess, QAFailure
    from .result_capture import capture_result
    live_access.require_local_request(budget.root, backend=spec.backend, graph=spec.profile == GRAPH)
    if (not isinstance(budget, PersistentBudget) or spec.profile not in (LINEAR, GRAPH)
            or budget.before_request is not live_access.require_local_access):
        raise BudgetError()

    def execute_once():
        with budget.reserve(spec, diagnostic_approval=diagnostic_approval) as reservation:
            reservation.check_setup(budget, profile=spec.profile, backend=spec.backend)
            live_access.require_local_request(budget.root, backend=spec.backend, graph=spec.profile == GRAPH)
            options = dict(backend=spec.backend, model=MODEL_ID, budget=budget,
                           reservation=reservation, allow_paid=spec.profile == LINEAR)
            if spec.profile == GRAPH:
                options['profile'] = GRAPH
            with opener(budget.root, **options) as service:
                result = service.ask(spec.question, mode=spec.mode, top_k=5,
                    reference_date=spec.reference_date, limits=limits, reservation=reservation)
            if not isinstance(result, (QASuccess, QAFailure)):
                raise BudgetError()
        return result
    return capture_result(execute_once, root=budget.root, invocation_id=spec.invocation_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approve-local-live', action='store_true', required=True,
                        help='Authorize paid use on this trusted single-user Mac, without tunneling or forwarding')
    parser.add_argument('--backend', choices=('chroma', 'pinecone'), default='chroma')
    parser.add_argument('--development', action='store_true', help='Select the separately approved monthly local development authority')
    parser.add_argument('--enable-graph', action='store_true', help='Separately opt into the seven-call semantic workflow')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args(argv)
    previous_tempdir = tempfile.tempdir
    try:
        if sys.version_info[:2] != (3, 12):
            raise ValueError('Python version')
        root = validate_project_root(Path.cwd())
        from .local_development import selected as development_selected, ENV
        if args.development:
            if args.enable_graph:
                raise BudgetError()
            os.environ[ENV] = '1'
        elif development_selected():
            raise BudgetError()
        for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
            if not contained_path(root, relative).is_dir():
                raise ValueError('Required project-local directory is unavailable')
        # Process-only settings. No .env, credentials or historical accounting is edited.
        os.environ.update(NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
            NORTHSTAR_UI_ENABLE_PINECONE=str(int(args.backend == 'pinecone')),
            NORTHSTAR_UI_ENABLE_GRAPH=str(int(args.enable_graph)), VECTOR_STORE=args.backend,
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            TMPDIR=str(root/'.local/tmp'), XDG_CACHE_HOME=str(root/'.local/cache'))
        tempfile.tempdir = str(root/'.local/tmp')
        with live_access.local_session(root, backend=args.backend, port=args.port,
                                       enable_graph=args.enable_graph, approved=args.approve_local_live):
            check_budget(root, enable_graph=args.enable_graph, backend=args.backend)
            from streamlit.web import bootstrap
            flags = server_options(args.port)
            bootstrap.load_config_options(flags)
            live_access.require_local_server()
            bootstrap.run(str(contained_path(root, 'src/northstar_atlas/ui/app.py')), False, [], flags)
        return 0
    except (BudgetError, ValueError, OSError, ImportError):
        print('Local live startup denied: verify authorization, existing authority, current pricing and local configuration.', file=sys.stderr)
        return 1
    finally:
        tempfile.tempdir = previous_tempdir


if __name__ == '__main__':
    raise SystemExit(main())
