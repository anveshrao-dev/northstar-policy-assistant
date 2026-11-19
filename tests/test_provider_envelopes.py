"""HTTP 200 envelope failures are not policy abstentions; no live API traffic."""
from copy import deepcopy
import unittest

import httpx
import test_provider_diagnostics as fixtures
import test_qa as qa_fixtures
from northstar_atlas.nebius import ProviderError


class ProviderEnvelopeRegressionTests(unittest.TestCase):
    def test_http_200_shape_content_model_and_finish_failures_are_withheld(self):
        cases = [None, [], {}, {'model': 'different-model', 'choices': []}]
        for choices in (None, [], [{}, {}], [{}], [{'message': None}], [{'message': []}]):
            value = fixtures.envelope(); value['choices'] = choices; cases.append(value)
        for content in (None, '', '   ', 7, [], {}):
            value = fixtures.envelope(); value['choices'][0]['message']['content'] = content; cases.append(value)
        for finish in (None, 'length', 'content_filter', 'tool_calls'):
            value = fixtures.envelope(); value['choices'][0]['finish_reason'] = finish; cases.append(value)
        for name in ('content', 'role'):
            value = fixtures.envelope(); del value['choices'][0]['message'][name]; cases.append(value)
        value = fixtures.envelope(); value['model'] = 'different-model'; cases.append(value)
        for index, value in enumerate(cases):
            with self.subTest(case=index):
                fixture = fixtures.ProviderDiagnosticsTests('runTest'); fixture.setUp()
                try:
                    with self.assertRaises(ProviderError):
                        fixture.invoke(httpx.Response(200, json=value) if value is not None else httpx.Response(200, text='null'))
                    fixture.record('response_envelope_validation')
                    fixture.uncertain()
                    self.assertEqual(fixture.sent, ['POST'])
                finally:
                    fixture.doCleanups()

    def test_valid_envelope_does_not_make_invalid_generation_or_quotations_publishable(self):
        fixture = qa_fixtures.QAServiceTests('runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        service, _, provider, prepared = fixture.make()
        missing = qa_fixtures.authored_claims(prepared)
        missing['main_answer']['claims'][0]['support'] = []
        wrong_quote = qa_fixtures.authored_claims(prepared)
        wrong_quote['main_answer']['claims'][0]['support'][0]['quote'] = 'SECRET_REJECTED'
        for body in ('{"kind":"answer",', 'null', '[]', missing, wrong_quote):
            with self.subTest(body=type(body).__name__):
                provider.body = deepcopy(body)
                before = len(provider.calls)
                fixture.assert_error(fixture.ask(service), 'response_rejected')
                self.assertEqual(len(provider.calls), before+1)
