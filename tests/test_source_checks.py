"""Dependency-free negative checks for the public integrity job."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceCheckTests(unittest.TestCase):
    def setUp(self):
        temp = ROOT / '.local/tmp'
        self.assertEqual(temp.resolve(), temp)
        temp.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=temp)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ('scripts', 'tests', 'src/northstar_atlas'):
            (self.root / name).mkdir(parents=True)
        for name in ('pyproject.toml', 'uv.lock'):
            shutil.copyfile(ROOT / name, self.root / name)
        checker = ROOT / 'scripts/check_source.py'
        if checker.exists():
            shutil.copyfile(checker, self.root / 'scripts/check_source.py')
        (self.root / 'src/northstar_atlas/__init__.py').write_text('')
        (self.root / 'tests/test_sample.py').write_text('')

    def run_check(self):
        return subprocess.run([sys.executable, '-I', '-S', '-B', str(self.root / 'scripts/check_source.py')],
                              cwd=self.root, capture_output=True, text=True)

    def test_valid_source_checks_without_imports_or_writes(self):
        (self.root / 'src/northstar_atlas/example.py').write_text(
            'raise RuntimeError("source must not execute")\n')
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_python_is_rejected(self):
        (self.root / 'src/northstar_atlas/broken.py').write_text('return 1\n')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python syntax', result.stderr)

    def test_changed_direct_dependency_pin_is_rejected(self):
        for old, new in [('torch==2.2.2', 'torch==0.0.0'), ('langgraph==1.2.16', 'langgraph==0.0.0'),
                         ('streamlit==1.66.1', 'streamlit==0.0.0')]:
            with self.subTest(dependency=old):
                text = (ROOT / 'pyproject.toml').read_text().replace(old, new)
                (self.root / 'pyproject.toml').write_text(text)
                result = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('dependency pin', result.stderr)

    def test_missing_package_is_rejected(self):
        (self.root / 'src/northstar_atlas/__init__.py').unlink()
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('package layout', result.stderr)

    def test_source_symlink_is_rejected(self):
        (self.root / 'src/northstar_atlas/alias.py').symlink_to('..')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source path', result.stderr)

    def test_symlinked_source_directory_cannot_hide_invalid_python(self):
        target = self.root / 'hidden'
        target.mkdir()
        (target / 'broken.py').write_text('return 1\n')
        (self.root / 'src/alias').symlink_to(target, target_is_directory=True)
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source path', result.stderr)


if __name__ == '__main__':
    unittest.main()
