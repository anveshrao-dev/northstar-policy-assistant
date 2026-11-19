"""Stdlib-only source integrity checks; no imports, builds or dependency solving."""
from pathlib import Path
import re
import sys
import tomllib


def check(root: Path) -> tuple[int, int]:
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    lock = tomllib.loads((root / 'uv.lock').read_text())
    if (project['project']['name'] != 'northstar-atlas'
            or not (root / 'src/northstar_atlas/__init__.py').is_file()):
        raise ValueError('package layout is missing or incompatible')
    if (project['project']['requires-python'].replace(' ', '') != '>=3.12,<3.13'
            or lock['requires-python'] != '==3.12.*'):
        raise ValueError('Python 3.12 manifest/lock contract differs')
    normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    locked = {(normalize(p['name']), p['version']) for p in lock['package']}
    # uv.lock describes application dependencies, not the isolated build backend.
    requirements = list(project['project']['dependencies'])
    for extra in project['project'].get('optional-dependencies', {}).values():
        requirements.extend(extra)
    for requirement in requirements:
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', requirement)
        if not match or (normalize(match[1]), match[2]) not in locked:
            raise ValueError('direct dependency pin is not present in uv.lock')
    count = 0
    for directory in ('src', 'scripts', 'tests'):
        folder = root / directory
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('source path is missing or unsafe')
        for path in sorted(folder.rglob('*')):
            # rglob does not descend into directory symlinks. Reject them too,
            # so a linked package cannot silently disappear from this check.
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('source path is unsafe')
            if path.suffix != '.py':
                continue
            if not path.is_file():
                raise ValueError('source path is unsafe')
            try:
                # Compile without executing, importing the package or creating pyc files.
                compile(path.read_bytes(), str(path.relative_to(root)), 'exec', dont_inherit=True)
            except SyntaxError:
                raise ValueError(f'Python syntax is invalid: {path.relative_to(root)}') from None
            count += 1
    return count, len(requirements)


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print('ERROR: source checks require Python 3.12', file=sys.stderr)
        return 1
    try:
        count, pins = check(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, KeyError, TypeError):
        # Public files only; report a fixed failure plus the deliberate validation reason.
        error = sys.exception()
        reason = str(error) if type(error) is ValueError else 'invalid source or manifest structure'
        print(f'ERROR: {reason}', file=sys.stderr)
        return 1
    print(f'PASS: {count} Python files compile; package layout and {pins} direct dependency pins match')
    print('Scope: static integrity only; not a package build, lock resolution or runtime test.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
