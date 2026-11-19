"""Run the installed project's unittest suite with Python socket access blocked.

This guard also loads in multiprocessing spawn workers. It is a regression-test
guard, not an OS network sandbox for arbitrary subprocess executables.
"""
import os
from pathlib import Path
import socket
import sys
import unittest


def denied(*args, **kwargs):
    raise OSError('network disabled for offline validation')


socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied


if __name__ == '__main__':
    from northstar_atlas.project import contained_path, validate_project_root
    root = validate_project_root(Path.cwd())
    temporary = contained_path(root, '.local/tmp')
    if not temporary.is_dir():
        raise SystemExit('Create the project-local .local/tmp directory before running tests.')
    os.environ.update(TMPDIR=str(temporary), PYTHONDONTWRITEBYTECODE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    sys.dont_write_bytecode = True
    import tempfile
    tempfile.tempdir = str(temporary)
    suite = unittest.defaultTestLoader.discover('tests', pattern=sys.argv[1] if len(sys.argv) > 1 else 'test*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f'COUNTS run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}')
    for _, reason in result.skipped:
        print('SKIP:', reason)
    raise SystemExit(not result.wasSuccessful())
