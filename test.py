import unittest
from app import app, model, cleanSolar

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_route(self):
        input_data = {'SystemSizeID': '1', 'SystemTypeID': '1', 'SolarPanelID': '1', 'BatteryID': '1', 'InverterID': '1'}
        response = self.app.post('/predict', json=input_data)
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_cost', data)
        self.assertIsInstance(data['predicted_cost'], str)
        self.assertTrue(float(data['predicted_cost']) >= 0.0)

    def test_recommend_route(self):
        input_text = 'solar power'
        response = self.app.post('/recommend', json={'input_text': input_text})
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn('recommendations', data)
        self.assertIsInstance(data['recommendations'], list)
        self.assertTrue(len(data['recommendations']) > 0)

    def test_invalid_input_predict_route(self):
        invalid_input = {'invalid_key': 'value'}
        response = self.app.post('/predict', json=invalid_input)
        data = response.get_json()

        self.assertEqual(response.status_code, 500)
        self.assertIn('error', data)
        self.assertIsInstance(data['error'], str)

    def test_invalid_input_recommend_route(self):
        invalid_input = {'invalid_key': 'value'}
        response = self.app.post('/recommend', json=invalid_input)
        data = response.get_json()

        self.assertEqual(response.status_code, 500)
        self.assertIn('error', data)
        self.assertIsInstance(data['error'], str)

if __name__ == '__main__':
    unittest.main()
