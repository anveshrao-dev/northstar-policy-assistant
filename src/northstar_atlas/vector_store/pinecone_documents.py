"""Read-only Pinecone Documents REST (2026-07), one attempt per HTTP operation.

Wire contract: https://docs.pinecone.io/reference/api/2026-07/data-plane/search_documents
(and fetch_documents / list_documents). No SDK internals or mutation endpoints.
Document fields survive intact for PineconeStore's frozen-record validation.
An ambiguous transport failure does not establish whether the server processed it.
"""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import math
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from .records import VectorStoreError


REASONS = {
    'json_malformed': 'json', 'json_duplicate_key': 'json', 'json_nonfinite_number': 'json',
    'envelope_type': 'envelope', 'envelope_missing_fields': 'envelope',
    'envelope_extra_fields': 'envelope', 'namespace_mismatch': 'envelope',
    'usage_type': 'usage', 'usage_missing_fields': 'usage', 'usage_extra_fields': 'usage', 'usage_read_units': 'usage',
    'usage_egress_bytes': 'usage',
    'pagination_fields': 'pagination', 'pagination_token': 'pagination',
    'pagination_repeated_token': 'pagination', 'pagination_limit': 'pagination',
    'document_list_type': 'document', 'document_list_count': 'document',
    'document_list_fields': 'document', 'document_id': 'document',
    'fetch_continuation': 'pagination', 'fetch_documents_type': 'document',
    'fetch_unexpected_ids': 'document', 'fetch_record_identity': 'document', 'search_records': 'document',
    'transport_timeout': 'transport', 'transport_failure': 'transport', 'http_rejection': 'http',
}


@dataclass(frozen=True)
class DocumentsDiagnostic:
    """Only fixed codes and validated numbers; never provider strings or objects."""

    operation: str
    validation_reason: str
    http_status: int | None
    ordinal: int
    list_attempts: int
    fetch_attempts: int
    search_attempts: int
    validated_read_units: int | None
    responses_with_validated_usage: int
    validated_egress_bytes: int | None = None
    responses_with_validated_egress: int = 0

    def __post_init__(self):
        if (type(self.operation) is not str or self.operation not in ('list', 'fetch')
                or type(self.validation_reason) is not str or self.validation_reason not in REASONS
                or (self.http_status is not None and
                    (type(self.http_status) is not int or not 100 <= self.http_status <= 599))
                or type(self.ordinal) is not int or self.ordinal < 1
                or any(type(n) is not int or n < 0 for n in
                       (self.list_attempts, self.fetch_attempts, self.search_attempts, self.responses_with_validated_usage))
                or (self.validated_read_units is not None and
                    (type(self.validated_read_units) is not int or self.validated_read_units < 0))
                or (self.validated_read_units is None) != (self.responses_with_validated_usage == 0)
                or self.responses_with_validated_usage > self.list_attempts + self.fetch_attempts + self.search_attempts):
            raise ValueError('invalid sanitized Documents diagnostic')
        validate_egress(self.validated_egress_bytes, self.responses_with_validated_egress,
                        self.responses_with_validated_usage)

    def to_dict(self):
        self.__post_init__()
        total = self.list_attempts + self.fetch_attempts + self.search_attempts
        return {'operation': self.operation, 'validation_reason': self.validation_reason,
                'field_group': REASONS[self.validation_reason], 'http_status': self.http_status,
                'page_ordinal': self.ordinal if self.operation == 'list' else None,
                'batch_ordinal': self.ordinal if self.operation == 'fetch' else None,
                'http_transport_attempts': {'list': self.list_attempts, 'fetch': self.fetch_attempts, 'search': self.search_attempts},
                'validated_read_units': self.validated_read_units,
                'responses_with_validated_usage': self.responses_with_validated_usage,
                'validated_egress_bytes': self.validated_egress_bytes,
                'responses_with_validated_egress': self.responses_with_validated_egress,
                'egress_complete': total > 0 and self.responses_with_validated_egress == total,
                'usage_complete': total > 0 and total == self.responses_with_validated_usage}


def validate_egress(total, responses, maximum_responses):
    """Validate sanitized aggregates; missing coverage never means zero bytes."""
    if (type(responses) is not int or not 0 <= responses <= maximum_responses
            or (total is None) != (responses == 0)
            or (total is not None and (type(total) is not int or not 0 <= total <= responses * 9223372036854775807))):
        raise ValueError('invalid sanitized egress usage')


class DocumentsError(RuntimeError):
    """Fixed, sanitized transport/configuration failure; never a provider body."""

    def __init__(self, message, *, reason=None):
        super().__init__(message)
        self.reason = reason


class _InvalidResponse(VectorStoreError):
    def __init__(self, reason):
        if reason not in REASONS:
            raise ValueError('invalid Documents validation reason')
        self.reason = reason
        super().__init__('invalid Pinecone Documents response; no partial evidence accepted')


def _invalid(reason):
    raise _InvalidResponse(reason) from None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid('json_duplicate_key')
        result[key] = value
    return result


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        _invalid('json_nonfinite_number')
    return number


@dataclass(repr=False)
class _Document:
    data: dict

    def to_dict(self):
        return deepcopy(self.data)


class _Pages:
    def __init__(self, documents, namespace, limit, timeout):
        self.documents, self.namespace = documents, namespace
        self.limit, self.timeout = limit, timeout

    def pages(self):
        token, seen = None, set()
        with self.documents._capture('list', 1) as context:
            # PineconeStore additionally bounds pages by the frozen record count.
            for page in range(1, 1025):
                context['ordinal'] = page
                body = {'limit': self.limit}
                if token is not None:
                    body['pagination_token'] = token
                value = self.documents._post('list', self.namespace, body, self.timeout, context=context)
                records = value['documents']
                if not isinstance(records, list): _invalid('document_list_type')
                if len(records) > self.limit: _invalid('document_list_count')
                if any(not isinstance(r, dict) or set(r) != {'_id'} for r in records):
                    _invalid('document_list_fields')
                if any(not isinstance(r['_id'], str) or not r['_id'] for r in records):
                    _invalid('document_id')
                token = self.documents._next(value)
                if token in seen:
                    _invalid('pagination_repeated_token')
                yield SimpleNamespace(items=[SimpleNamespace(id=r['_id']) for r in records], pagination_token=token)
                if token is None:
                    return
                seen.add(token)
            _invalid('pagination_limit')


class _CountedTransport(httpx.BaseTransport):
    """Observe dispatch, not logical call setup or assumed server processing."""
    def __init__(self, transport):
        self.transport = transport
        self.attempts = {'list': 0, 'fetch': 0, 'search': 0}

    def handle_request(self, request):
        operation = request.url.path.rsplit('/', 1)[-1]
        if operation not in self.attempts:
            raise DocumentsError('unsupported Pinecone Documents operation')
        self.attempts[operation] += 1
        return self.transport.handle_request(request)

    def close(self):
        self.transport.close()


class DocumentsIndex:
    """Context-owned read-only transport selected only after target confirmation.

    ``transport`` is dependency injection for offline tests. Production always
    constructs HTTPTransport(retries=0), without proxy or environment overrides.
    The existing SDK control-plane context suppresses HTTP diagnostics; the same
    save/restore protection also covers standalone calls to this transport.
    """

    def __init__(self, *, host, api_key, observe=None, transport=None, on_failure=None, observe_egress=None):
        parsed = urlsplit(host) if isinstance(host, str) else None
        if (parsed is None or parsed.scheme != 'https' or not parsed.hostname
                or not re.fullmatch(r'[a-z0-9][a-z0-9.-]*\.pinecone\.io', parsed.hostname)
                or parsed.netloc != parsed.hostname or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment):
            raise DocumentsError('invalid Pinecone Documents target')
        if not isinstance(api_key, str) or not api_key or not api_key.isascii() or any(c.isspace() for c in api_key):
            raise DocumentsError('invalid Pinecone Documents credential configuration')
        self._host = host.rstrip('/')
        self._observe = observe
        self._observe_egress = observe_egress
        self._on_failure = on_failure
        self._fetch_ordinal = 0
        self._read_units = 0
        self._usage_responses = 0
        self._egress_bytes = 0
        self._egress_responses = 0
        self._transport = _CountedTransport(transport if transport is not None else httpx.HTTPTransport(retries=0, trust_env=False))
        self._client = httpx.Client(
            transport=self._transport,
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(connect=5., read=10., write=10., pool=5.),
            headers={'Api-Key': api_key, 'X-Pinecone-Api-Version': '2026-07', 'Content-Type': 'application/json'},
        )
        self.documents = self

    @contextmanager
    def _capture(self, operation, ordinal):
        context = {'ordinal': ordinal, 'http_status': None}
        try:
            yield context
        except (_InvalidResponse, DocumentsError) as exc:
            if self._on_failure is not None and exc.reason in REASONS:
                counts = self._transport.attempts
                self._on_failure(DocumentsDiagnostic(operation, exc.reason, context['http_status'],
                    context['ordinal'], counts['list'], counts['fetch'], counts['search'],
                    self._read_units if self._usage_responses else None, self._usage_responses,
                    self._egress_bytes if self._egress_responses else None, self._egress_responses))
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._client.close()

    @staticmethod
    def _namespace(namespace):
        if not isinstance(namespace, str) or not re.fullmatch(r'dev-[0-9a-f]{64}', namespace):
            raise DocumentsError('invalid Pinecone Documents namespace')

    @staticmethod
    def _next(value):
        pagination = value.get('pagination')
        if pagination is None:
            return None
        if not isinstance(pagination, dict) or set(pagination) - {'next'}:
            _invalid('pagination_fields')
        token = pagination.get('next')
        if token is not None and (not isinstance(token, str) or not token or len(token) > 16384):
            _invalid('pagination_token')
        return token

    def _post(self, operation, namespace, body, timeout, *, context=None):
        self._namespace(namespace)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 10:
            raise DocumentsError('invalid Pinecone Documents timeout')
        if operation not in ('list', 'fetch', 'search'):
            raise DocumentsError('unsupported Pinecone Documents operation')
        if self._observe is not None:
            self._observe(operation, None)
        if context is not None: context['http_status'] = None
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            # One send, no redirect following, status retry, reauthentication or recovery.
            response = self._client.post(f'{self._host}/namespaces/{namespace}/documents/{operation}',
                json=body, timeout=httpx.Timeout(connect=min(5., timeout), read=timeout,
                                                write=timeout, pool=min(5., timeout)))
        except httpx.TimeoutException:
            raise DocumentsError('Pinecone Documents transport timeout; reconcile before retrying', reason='transport_timeout') from None
        except VectorStoreError:
            # A diagnostic dispatch gate can refuse I/O before transport. Keep
            # that local refusal distinct from an attempted HTTP failure.
            raise
        except (httpx.HTTPError, ValueError):
            raise DocumentsError('Pinecone Documents transport failed; reconcile before retrying', reason='transport_failure') from None
        finally:
            logging.disable(previous)
        if context is not None: context['http_status'] = response.status_code
        if response.status_code != 200:
            raise DocumentsError('Pinecone Documents HTTP rejection; no retry', reason='http_rejection')
        try:
            value = json.loads(response.content, object_pairs_hook=_object, parse_float=_float,
                               parse_constant=lambda _: _invalid('json_nonfinite_number'))
        except _InvalidResponse:
            raise
        except (ValueError, UnicodeError, RecursionError):
            _invalid('json_malformed')
        field = 'matches' if operation == 'search' else 'documents'
        required = {'namespace', 'usage', field}
        optional = set() if operation == 'search' else {'pagination'}
        if not isinstance(value, dict): _invalid('envelope_type')
        if not required <= set(value): _invalid('envelope_missing_fields')
        if set(value) - required - optional: _invalid('envelope_extra_fields')
        if value['namespace'] != namespace: _invalid('namespace_mismatch')
        usage = value['usage']
        if not isinstance(usage, dict): _invalid('usage_type')
        if 'read_units' not in usage: _invalid('usage_missing_fields')
        # Search supports precomputed dense vectors only: integrated-embedding
        # usage (embed_total_tokens) is incompatible with this request profile.
        allowed_usage = {'read_units', 'egress_bytes'}
        if set(usage) - allowed_usage: _invalid('usage_extra_fields')
        if type(usage['read_units']) is not int or not 0 <= usage['read_units'] <= 2147483647:
            _invalid('usage_read_units')
        if 'egress_bytes' in usage and (type(usage['egress_bytes']) is not int
                or not 0 <= usage['egress_bytes'] <= 9223372036854775807):
            _invalid('usage_egress_bytes')
        # Commit only after the entire usage object passes validation.
        self._read_units += usage['read_units']
        self._usage_responses += 1
        if 'egress_bytes' in usage:
            self._egress_bytes += usage['egress_bytes']
            self._egress_responses += 1
        if self._observe is not None:
            self._observe(operation, usage['read_units'])
        if self._observe_egress is not None:
            self._observe_egress(usage.get('egress_bytes'))
        return value

    def list(self, *, namespace, limit=100, timeout=10):
        self._namespace(namespace)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise DocumentsError('invalid Pinecone Documents list limit')
        return _Pages(self, namespace, limit, timeout)

    def fetch(self, *, namespace, ids, timeout=10):
        if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 1000
                or any(not isinstance(i, str) or not i or len(i) > 512 for i in ids)
                or len(set(ids)) != len(ids)):
            raise DocumentsError('invalid Pinecone Documents fetch identifiers')
        self._fetch_ordinal += 1
        with self._capture('fetch', self._fetch_ordinal) as context:
            value = self._post('fetch', namespace, {'ids': list(ids)}, timeout, context=context)
            records = value['documents']
            if self._next(value) is not None: _invalid('fetch_continuation')
            if not isinstance(records, dict): _invalid('fetch_documents_type')
            if set(records) - set(ids): _invalid('fetch_unexpected_ids')
            if any(not isinstance(r, dict) or r.get('_id') != key for key, r in records.items()):
                _invalid('fetch_record_identity')
            return SimpleNamespace(namespace=namespace, documents={k: _Document(r) for k, r in records.items()})

    def search(self, *, namespace, top_k, score_by, include_fields, filter, timeout=10):
        if (type(top_k) is not int or not 1 <= top_k <= 10000
                or not isinstance(score_by, list) or len(score_by) != 1
                or not isinstance(score_by[0], dict)
                or set(score_by[0]) != {'type', 'field', 'values'}
                or score_by[0]['type'] != 'dense_vector' or score_by[0]['field'] != 'embedding'):
            raise DocumentsError('invalid Pinecone Documents dense query')
        vector = score_by[0]['values']
        if (not isinstance(vector, list) or not vector
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                or not isinstance(include_fields, list) or not include_fields
                or any(not isinstance(f, str) or not f for f in include_fields)
                or len(set(include_fields)) != len(include_fields)
                or not isinstance(filter, dict) or not filter):
            raise DocumentsError('invalid Pinecone Documents dense query')
        value = self._post('search', namespace, {'top_k': top_k, 'score_by': score_by,
                           'include_fields': include_fields, 'filter': filter}, timeout)
        matches = value['matches']
        if (not isinstance(matches, list) or len(matches) > top_k
                or any(not isinstance(r, dict) or not isinstance(r.get('_id'), str)
                       or not r['_id'] or type(r.get('_score')) not in (int, float)
                       or not math.isfinite(r['_score']) for r in matches)):
            _invalid('search_records')
        return SimpleNamespace(namespace=namespace, matches=[_Document(r) for r in matches])
