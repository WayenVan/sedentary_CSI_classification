import unittest
from csi_catm.data.common import parse_catm_file_name

class TestDataset(unittest.TestCase):
    
    def test_3channel(self):
        file_name = "user12-1-2-42.m"
        user, action, channel, index = parse_catm_file_name(file_name)
        self.assertEqual(user, 12)
        assert action == 1
        assert channel == 2
        assert index == 42


if __name__ == "__main__":
    unittest.main()