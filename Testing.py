import unittest
from PAM import PAM


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_determine_closest_in_dictionary(self):
        test_dict = {
            "1": 55, "2": 22, "3": 11, "4": 1
        }
        result_list = PAM.order_by_dict_values(test_dict)
        result = result_list[0][1]
        self.assertEqual(result, 1, "Works")

if __name__ == '__main__':
    unittest.main()
