import unittest
import main
import app
import requests

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
        self.assertNotEqual(main.saveAudio('https://www.youtube.com/watch?v=Zbsadhfajb'), True)
        # valid video of over 10m
        self.assertNotEqual(main.saveAudio('https://www.youtube.com/watch?v=God7bXyKkdA&t'), True)

    # test saveMFCC
    def test_saveMFCC(self):
        self.assertEqual(main.saveMFCC(num_segments=5, hop_length=512, n_mfcc=13, n_fft=2048), True)

class TestApp(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.app = app.app.test_client()

    def test_home(self):
        # homepage is served
        result = self.app.get('/')
        assert b'Classifier' in result.data

    def test_predict(self):
        # get not allowed
        result = self.app.get('/predict')
        assert b'405' in result.data

        # post request test
        result = self.app.post('/predict', data={"link":"https://www.youtube.com/watch?v=ZbZSe6N_BXs"})
        print('result is: {}'.format(result))
        # TODO change this so it asserts correct functionality - 200?
        assert b'400' not in result.data

        # invalid link error message gets passed to frontend
        result = self.app.post('/predict', data={"link": "https://www.youtube.com/watch?v=Zb_BXs"})
        print('result is: {}'.format(result))
        assert b'recognised' in result.data

        # too long video error gets passed to frontend
        result = self.app.post('/predict', data={"link": "https://www.youtube.com/watch?v=FDMq9ie0ih0"})
        print('result is: {}'.format(result))
        assert b'15 minutes' in result.data

if __name__ == '__main__':
    unittest.main()
