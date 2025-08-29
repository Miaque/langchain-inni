import unittest
import uuid

from tools.sandbox.sandbox import create_sandbox


class SandboxTestCase(unittest.TestCase):
    def test_run_sandbox(self):
        import asyncio

        asyncio.run(create_sandbox(str(uuid.uuid4())))


if __name__ == "__main__":
    unittest.main()
