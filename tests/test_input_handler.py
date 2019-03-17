from unittest import TestCase
import common.input_handler as uin


class TestInputHandler(TestCase):
    def test_get_input_int(self):
        expected_output = [1, 2, 3, 4, 5]

        output = uin.get_input_int(msg="Enter number(1..5): ", only_accept=expected_output, error_msg="not valid")
        if isinstance(output, int) and output in expected_output:
            self.assertTrue(True)
        else:
            self.fail()

    def test_get_input_str(self):
        expected_output = ['train', 'test']

        output = uin.get_input_str(msg="Enter str(test/train): ", only_accept=expected_output, error_msg="not valid")
        if isinstance(output, str) and output in expected_output:
            self.assertTrue(True)
        else:
            self.fail()

    def test_get_confirmation(self):
        output = uin.get_confirmation(msg="Please confirm: ", error_msg="not valid")
        print(type(output))

        if isinstance(output, bool):
            self.assertTrue(True)
        else:
            self.fail()
