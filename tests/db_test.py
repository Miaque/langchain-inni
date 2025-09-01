import unittest

from loguru import logger
from sqlalchemy import text

from storage.db import get_db


class DatabaseTestCase(unittest.TestCase):
    def test_conn(self):
        with get_db() as db:
            result = db.execute(text("select TIMEZONE('Asia/Shanghai'::text, NOW())"))
            row = result.fetchone()
            logger.info(row)


if __name__ == "__main__":
    unittest.main()
