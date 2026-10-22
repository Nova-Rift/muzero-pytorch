import tempfile
import unittest
from pathlib import Path
from arena.state_history import save_milestone


class StateHistoryTests(unittest.TestCase):
    def test_milestones_survive_latest_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            def save(state, path):
                path.write_bytes(str(state['totals']['games']).encode())
            first = save_milestone({'totals': {'games': 512, 'optimizer_updates': 512}}, out, save)
            second = save_milestone({'totals': {'games': 1024, 'optimizer_updates': 1024}}, out, save)
            self.assertEqual(first.read_bytes(), b'512')
            self.assertEqual(second.read_bytes(), b'1024')
            self.assertEqual((out/'resume-state.pt').read_bytes(), b'1024')
            with self.assertRaises(FileExistsError):
                save_milestone({'totals': {'games': 512, 'optimizer_updates': 512}}, out, save)
            self.assertEqual((out/'resume-state.pt').read_bytes(), b'1024')

    def test_partial_serialization_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            save_milestone({'totals': {'games': 1, 'optimizer_updates': 1}}, out,
                           lambda state, path: path.write_bytes(b'complete'))
            def fail(state, path):
                path.write_bytes(b'partial')
                raise OSError('disk full')
            with self.assertRaises(OSError):
                save_milestone({'totals': {'games': 2, 'optimizer_updates': 2}}, out, fail)
            self.assertEqual((out/'resume-state.pt').read_bytes(), b'complete')
            self.assertEqual(len(list((out/'full-state').iterdir())), 1)
