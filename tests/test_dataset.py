import unittest
from csi_catm.data.common import parse_catm_file_name, aggregate_3channel

import os

class TestDataset(unittest.TestCase):
    
    def test_3channel(self):
        file_name = "user12-1-2-42.m"
        user, action, channel, index = parse_catm_file_name(file_name)
        self.assertEqual(user, 12)
        self.assertEqual(action, 1)
        self.assertEqual(channel, 2)
        self.assertEqual(index, 42)
        
        file_list = os.listdir("dataset/CATM")
        ret = aggregate_3channel(file_list)
        for item in ret:
            for idx, name in enumerate(item):
                user, action, channel,index = parse_catm_file_name(name)
                self.assertEqual(channel, idx+1)
            self.assertEqual(len(item), 3)

if __name__ == "__main__":
    unittest.main()