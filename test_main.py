import unittest
import main

class TestMain(unittest.TestCase):

    # url validation tests
    # invalid url passed, valid, empty, leading space, non-YT link
    def test_validate(self):
        # invalid link
        self.assertNotEqual(main.validate('invalid'), True)
        # valid link
        self.assertEqual(main.validate('https://youtube.com'), True)
        # empty string
        self.assertNotEqual(main.validate(''), True)
        # leading space
        self.assertNotEqual(main.validate(' https://youtube.com'), True)
        # non-YouTube link
        self.assertNotEqual(main.validate('https://google.com'), True)

    # test YouTube audio conversion
    def test_saveAudio(self):
        # valid song of < 10m
        self.assertEqual(main.saveAudio('https://www.youtube.com/watch?v=ZbZSe6N_BXs'), True)
        # invalid link
        self.assertEqual(main.saveAudio('https://www.youtube.com/watch?v=Zbsadhfajb'), False)
        # valid video of over 10m
        self.assertEqual(main.saveAudio('https://www.youtube.com/watch?v=God7bXyKkdA&t'), False)

    # test saveMFCC
    def test_saveMFCC(self):
        self.assertEqual(main.saveMFCC(num_segments=5, hop_length=512, n_mfcc=13, n_fft=2048), True)

if __name__ == '__main__':
    unittest.main()