"""Separate local operator authorization from hosted Private Preview verification.

Only the dedicated in-process Mac bootstrap can select the loopback policy.
Environment flags alone never replace the hosted access verifier.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import platform

from .project import validate_project_root


class LocalAccessError(ValueError):
    """Fixed safe failure; no request headers or configuration values are exposed."""


@dataclass(frozen=True)
class _LocalLaunch:
    root: Path
    backend: str
    port: int
    enable_graph: bool


_local: _LocalLaunch | None = None
_HOSTED_ENV = ('REPLIT_DEV_DOMAIN', 'REPL_ID', 'REPL_OWNER', 'REPL_SLUG', 'REPLIT_DEPLOYMENT')


def _deny():
    raise LocalAccessError('Local live access is not authorized.')


def _static_check(launch):
    if (launch is None or platform.system() != 'Darwin'
            or any(name in os.environ for name in _HOSTED_ENV)
            or os.environ.get('NORTHSTAR_CONTROLLED_LIVE') != '1'
            or os.environ.get('NORTHSTAR_UI_ENABLE_LIVE') != '1'
            or os.environ.get('VECTOR_STORE') != launch.backend
            or os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH', '0') != str(int(launch.enable_graph))
            or os.environ.get('NORTHSTAR_UI_ENABLE_PINECONE', '0') != str(int(launch.backend == 'pinecone'))
            or validate_project_root(launch.root) != launch.root):
        _deny()


@contextmanager
def local_session(root: Path, *, backend: str, port: int, enable_graph: bool, approved: bool):
    """Trusted bootstrap scope, never called with browser/question parameters."""
    global _local
    if (_local is not None or approved is not True or backend not in ('chroma', 'pinecone')
            or type(enable_graph) is not bool or type(port) is not int or not 1024 <= port <= 65535):
        _deny()
    launch = _LocalLaunch(validate_project_root(root), backend, port, enable_graph)
    _static_check(launch)
    _local = launch
    try:
        yield
    finally:
        _local = None


def local_backend() -> str | None:
    """Select the local path even after revocation, so it fails closed there."""
    return _local.backend if _local is not None else None


def operator_access_authorized() -> bool:
    if _local is None:
        return os.environ.get('NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED') == '1'
    try:
        _static_check(_local)
        return True
    except (ValueError, OSError):
        return False


def access_check():
    if _local is not None:
        return require_local_access
    from .live_preflight import require_private_preview
    return require_private_preview


def require_live_budget(root, budget) -> None:
    """Shared paid-entry gate, before reservation or resource construction.

    Flags express operator intent; they do not substitute for a coordinator or
    its local/hosted access verifier. No credential or ledger reads occur here.
    """
    from .live_budget import BudgetError, PersistentBudget
    if (os.environ.get('NORTHSTAR_CONTROLLED_LIVE') != '1'
            or not operator_access_authorized()
            or not isinstance(budget, PersistentBudget) or budget.root != root
            or budget.before_request is not access_check()):
        raise BudgetError()


def require_live_reservation(root, budget, reservation, *, profile, backend) -> None:
    """Check durable identity, policy, pricing and the selected access authority."""
    from .live_budget import BudgetError, WorkflowReservation, LINEAR, GRAPH
    require_live_budget(root, budget)
    if (not isinstance(reservation, WorkflowReservation) or backend not in ('chroma', 'pinecone')
            or profile not in (LINEAR, GRAPH)
            or (profile == GRAPH and os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH') != '1')
            or (_local is not None and (backend != _local.backend or (profile == GRAPH and not _local.enable_graph)))):
        raise BudgetError()
    reservation.check_setup(budget, profile=profile, backend=backend)


def require_local_server() -> None:
    """Recheck actual Streamlit configuration, not merely launch arguments."""
    _static_check(_local)
    from streamlit import config
    from .local_live import server_options
    if config.get_option('server.sslCertFile') or config.get_option('server.sslKeyFile'):
        _deny()
    for name, expected in server_options(_local.port).items():
        if config.get_option(name.replace('_', '.')) != expected:
            _deny()


def require_local_access() -> None:
    """Require a direct loopback browser session before resources or each call.

    Streamlit's public ip_address deliberately returns None for loopback. The
    pinned runtime's session client context is checked instead; absent contexts
    fail closed. This is a single-user Mac policy, not remote authentication.
    """
    try:
        require_local_server()
        from streamlit.runtime.context import _get_client_context
        client = _get_client_context()
        if client is None or client.remote_ip != '127.0.0.1':
            _deny()
        headers = list(client.headers)
        # The pinned ClientContext returns pairs, preserving duplicate headers.
        def values(name):
            return [v for k, v in headers if k.lower() == name.lower()]
        if (values('Host') != [f'127.0.0.1:{_local.port}']
                or values('Origin') != [f'http://127.0.0.1:{_local.port}']
                or any(k.lower() == 'forwarded' or k.lower().startswith(('x-forwarded-', 'x-real-'))
                       for k, _ in headers)):
            _deny()
    except (ImportError, AttributeError, TypeError, KeyError, RuntimeError, OSError):
        _deny()


def require_local_request(root: Path, *, backend: str, graph: bool) -> None:
    require_local_access()
    if root != _local.root or backend != _local.backend or (graph and not _local.enable_graph):
        _deny()
