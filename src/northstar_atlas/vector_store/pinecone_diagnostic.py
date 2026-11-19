"""Explicit incomplete-publication diagnosis. No readiness promotion or SDK calls.

REST describe contract: /reference/api/2026-07/control-plane/describe_index at
docs.pinecone.io. A dispatch count is not proof of server processing or billing.
HTTPX phase timeouts are not an overall elapsed-time deadline.
"""

from contextlib import ExitStack
import json
import logging
import re
from threading import Lock
from types import SimpleNamespace

import httpx

from .pinecone_cloud import CloudError, check_confirmation, load_settings, validate_description
from .pinecone_documents import DocumentsIndex, REASONS, _object, _float, validate_egress
from .pinecone_store import PineconeTarget
from . import pinecone_workflow as workflow
from .records import VectorStoreError


class DiagnosticError(VectorStoreError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('diagnostic audit stopped; inspect sanitized report')


class DispatchBudget:
    """One invocation, 12 physical transport dispatches, no refunds."""
    def __init__(self):
        self._lock = Lock()
        self._counts = {'describe': 0, 'list': 0, 'fetch': 0}
        self.last_operation = None

    def reserve(self, operation):
        with self._lock:
            if operation not in self._counts:
                raise DiagnosticError('unexpected_operation')
            # Identify a refused operation without pretending it was dispatched.
            self.last_operation = operation
            if sum(self._counts.values()) >= 12:
                raise DiagnosticError('budget_exhausted')
            self._counts[operation] += 1

    def counts(self):
        with self._lock:
            return dict(self._counts)


class BudgetTransport(httpx.BaseTransport):
    """Gate immediately before the no-retry transport, across both planes."""
    def __init__(self, budget, allowed, transport=None):
        self.budget, self.allowed = budget, dict(allowed)
        self.transport = transport if transport is not None else httpx.HTTPTransport(
            retries=0, trust_env=False, verify=True)

    def handle_request(self, request):
        operation = self.allowed.get((request.method, str(request.url)))
        if operation is None:
            raise DiagnosticError('unexpected_operation')
        self.budget.reserve(operation)
        return self.transport.handle_request(request)

    def close(self):
        self.transport.close()


class DescribeClient:
    def __init__(self, settings, budget, *, transport=None):
        if not isinstance(settings.name, str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?', settings.name):
            raise DiagnosticError('target_validation')
        self.settings = settings
        self.url = 'https://api.pinecone.io/indexes/' + settings.name
        self.http_status = None
        self.client = httpx.Client(transport=BudgetTransport(budget, {('GET', self.url): 'describe'}, transport),
            trust_env=False, follow_redirects=False, timeout=httpx.Timeout(connect=5., read=10., write=10., pool=5.),
            headers={'Api-Key': settings.api_key, 'X-Pinecone-Api-Version': '2026-07'})

    def __enter__(self): return self
    def __exit__(self, *_): self.client.close()

    def describe(self):
        self.http_status = None
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            try:
                response = self.client.get(self.url)
            except httpx.TimeoutException:
                raise DiagnosticError('transport_timeout') from None
            except (httpx.HTTPError, ValueError) as exc:
                if isinstance(exc, DiagnosticError): raise
                raise DiagnosticError('transport_failure') from None
        finally:
            logging.disable(previous)
        self.http_status = response.status_code
        if response.status_code != 200:
            raise DiagnosticError('http_rejection')
        try:
            value = json.loads(response.content, object_pairs_hook=_object, parse_float=_float,
                parse_constant=lambda _: (_ for _ in ()).throw(DiagnosticError('json_nonfinite_number')))
        except VectorStoreError as exc:
            reason = getattr(exc, 'reason', None)
            raise DiagnosticError(reason if reason in REASONS else 'json_malformed') from None
        except (ValueError, UnicodeError, RecursionError):
            raise DiagnosticError('json_malformed') from None
        try:
            host = value['host']
            if not isinstance(host, str): raise ValueError
            bare = host.removeprefix('https://')
            if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*\.pinecone\.io', bare): raise ValueError
            value['host'] = 'https://' + bare
            # Preserve safety-relevant raw fields; never infer a legacy vector schema.
            if not isinstance(value['status']['state'], str): raise ValueError
            field = value['schema']['fields']['embedding']
            if any(field.get(k) for k in ('embed', 'sparse_embed')): raise ValueError
            return validate_description(value, self.settings, require_ready=True)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise DiagnosticError('target_validation') from None


STAGES = {'confirmation', 'describe', 'snapshot', 'publication', 'ownership', 'source', 'audit_target', 'records', 'complete'}
EXTRA_REASONS = {'none', 'budget_exhausted', 'unexpected_operation', 'target_validation', 'confirmation',
    'describe', 'snapshot', 'publication', 'ownership', 'source', 'audit_target', 'record_integrity',
    'publication_changed', 'unexpected_failure', 'interrupted'}


def save_diagnostic_report(root, report):
    """Validate and project every field; never serialize arbitrary exception data."""
    keys = {'schema_version', 'outcome', 'operation', 'stage', 'reason', 'http_status',
            'page_ordinal', 'batch_ordinal', 'attempts', 'validated_read_units',
            'responses_with_validated_usage', 'publication_preserved',
            'validated_egress_bytes', 'responses_with_validated_egress', 'egress_complete'}
    if type(report) is not dict or set(report) != keys:
        raise ValueError('invalid diagnostic report')
    counts = report['attempts']
    if (type(report['schema_version']) is not int or report['schema_version'] != 2
            or report['outcome'] not in ('success', 'failed', 'inconclusive')
            or report['operation'] not in (None, 'describe', 'list', 'fetch')
            or report['stage'] not in STAGES or report['reason'] not in EXTRA_REASONS | REASONS.keys()
            or type(counts) is not dict or set(counts) != {'describe', 'list', 'fetch'}
            or any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) > 12
            or (report['publication_preserved'] is not None and type(report['publication_preserved']) is not bool)):
        raise ValueError('invalid diagnostic report')
    for name in ('http_status', 'page_ordinal', 'batch_ordinal', 'validated_read_units'):
        n = report[name]
        if n is not None and (type(n) is not int or n < (0 if name == 'validated_read_units' else 1)):
            raise ValueError('invalid diagnostic report')
    status = report['http_status']
    usage_count = report['responses_with_validated_usage']
    if (status is not None and not 100 <= status <= 599
            or type(usage_count) is not int or not 0 <= usage_count <= counts['list'] + counts['fetch']
            or (report['validated_read_units'] is None) != (usage_count == 0)):
        raise ValueError('invalid diagnostic report')
    validate_egress(report['validated_egress_bytes'], report['responses_with_validated_egress'], usage_count)
    complete = counts['list'] + counts['fetch'] > 0 and report['responses_with_validated_egress'] == counts['list'] + counts['fetch']
    if type(report['egress_complete']) is not bool or report['egress_complete'] != complete:
        raise ValueError('invalid diagnostic report')
    return workflow._save_private_report(root, 'diagnostic-audit', {k: report[k] for k in keys})


def _audit_incomplete(store, report):
    state = store.state
    state.check_paths()
    # Do not create a new publication directory for an unexpected described host.
    if not state.path.is_dir(): raise DiagnosticError('publication')
    with state.lock():
        paths = (state.path / (store.namespace + '.json'), state.path / 'active.json')
        before = tuple(p.read_bytes() for p in paths)
        try:
            run = state.read_run(store.namespace)
            if (run is None or run.get('state') != 'incomplete' or run.get('phase') != 'failed audit'
                    or state.active() != store.namespace):
                raise DiagnosticError('publication')
            report['stage'] = 'ownership'
            store._owned(run)
            report['stage'] = 'source'
            store._source_check()
            report['stage'] = 'audit_target'
            store._target_check()
            report['stage'] = 'records'
            # Unlike ordinary audit, no invalidation write and no _audited=True.
            store._audit()
            report['stage'] = 'source'
            store._source_check()
        finally:
            store._audited = False
            try:
                state.check_paths()
                report['publication_preserved'] = before == tuple(p.read_bytes() for p in paths)
            except Exception:
                report['publication_preserved'] = False
            if report['publication_preserved'] is not True:
                raise DiagnosticError('publication_changed') from None


def run_diagnostic(root, settings=None, *, control_transport=None, documents_transport=None):
    """Explicit operator entry point. Transport injection is for offline fixtures.

    Reports failure/inconclusive outcomes, but never retries or restores state.
    A report persistence error propagates to the CLI, which must stop.
    """
    budget = DispatchBudget()
    meter = workflow.RequestMeter()
    usage = {'units': 0, 'responses': 0}
    report = {'schema_version': 2, 'outcome': 'failed', 'operation': None, 'stage': 'confirmation',
        'reason': 'none', 'http_status': None, 'page_ordinal': None, 'batch_ordinal': None,
        'attempts': budget.counts(), 'validated_read_units': None, 'responses_with_validated_usage': 0,
        'publication_preserved': None}
    control = None

    def observe(operation, units):
        if units is not None:
            usage['units'] += units
            usage['responses'] += 1

    try:
        if settings is None:
            settings = load_settings(root)
        check_confirmation(root, settings)
        with ExitStack() as stack:
            report['stage'] = 'describe'
            control = stack.enter_context(DescribeClient(settings, budget, transport=control_transport))
            description = control.describe()
            target = PineconeTarget(settings.name, description['host'], settings.project_id, settings.cloud, settings.region)
            report['stage'] = 'snapshot'
            snapshot = workflow.active_snapshot(root, target)
            allowed = {('POST', f'{target.host}/namespaces/{snapshot.namespace}/documents/{op}'): op for op in ('list', 'fetch')}
            transport = BudgetTransport(budget, allowed, documents_transport)
            index = stack.enter_context(DocumentsIndex(host=target.host, api_key=settings.api_key,
                transport=transport, observe=observe, on_failure=meter.record_documents_failure,
                observe_egress=meter.observe_egress))
            # Reuse the existing 189-record/model/source validator and store constructor.
            client = SimpleNamespace(indexes=SimpleNamespace(describe=lambda **kwargs: control.describe()))
            store = workflow.open_store(root, snapshot, target, client, index)
            report['stage'] = 'publication'
            _audit_incomplete(store, report)
            report.update(outcome='success', stage='complete')
    except DiagnosticError as exc:
        report['reason'] = exc.reason
        if exc.reason == 'budget_exhausted': report['outcome'] = 'inconclusive'
    except (CloudError, VectorStoreError):
        report['reason'] = 'record_integrity' if report['stage'] == 'records' else report['stage']
    except KeyboardInterrupt:
        report.update(outcome='inconclusive', reason='interrupted')
    except Exception:
        report.update(outcome='inconclusive', reason='unexpected_failure')
    report.update(attempts=budget.counts(), operation=budget.last_operation,
        validated_read_units=usage['units'] if usage['responses'] else None,
        responses_with_validated_usage=usage['responses'],
        validated_egress_bytes=meter.egress_bytes if meter.egress_responses else None,
        responses_with_validated_egress=meter.egress_responses,
        egress_complete=budget.counts()['list'] + budget.counts()['fetch'] > 0 and
            meter.egress_responses == budget.counts()['list'] + budget.counts()['fetch'])
    if control is not None and budget.last_operation == 'describe':
        report['http_status'] = control.http_status
    if meter.documents_failure is not None and report['reason'] != 'publication_changed':
        evidence = meter.documents_failure.to_dict()
        report.update(reason=evidence['validation_reason'], http_status=evidence['http_status'],
                      page_ordinal=evidence['page_ordinal'], batch_ordinal=evidence['batch_ordinal'])
    save_diagnostic_report(root, report)
    return report
