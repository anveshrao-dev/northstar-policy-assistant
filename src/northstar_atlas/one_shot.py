"""Select an existing one-shot contract; never prepare or activate accounting."""
import os


def _controller():
    from . import authority_handoff, local_test_authority
    # Presence selects the restricted path. Mixed or malformed settings fail in
    # configured_plan rather than falling back to a generic/batch permission.
    return local_test_authority if local_test_authority.selected() else authority_handoff


def one_shot_selected():
    from . import authority_handoff, local_test_authority
    return local_test_authority.selected() or authority_handoff.one_shot_selected()


def configured_plan():
    return _controller().configured_plan()


def inspect_invocation(budget, plan_id, *, require_browser=True):
    return _controller().inspect_invocation(budget, plan_id, require_browser=require_browser)


def backend(status):
    # The existing immutable handoff contract is Pinecone-only.
    return getattr(status, 'backend', 'pinecone')


def run_question(budget, request, *, plan_sha256, consent, opener, limits):
    from . import local_test_authority
    controller = _controller()
    fresh = controller is local_test_authority
    execute = controller.run_question if fresh else controller.run_handoff_question
    setting = local_test_authority.ENV_RUN if fresh else 'NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL'
    return execute(budget, request, plan_sha256=plan_sha256,
        approved_run_sha256=os.environ.get(setting), consent=consent, opener=opener, limits=limits)
