"""
test_api.py — Flask API Tests
==============================
Run: python test_api.py
Note: app.py same folder mein hona chahiye aur model bhi.
"""

import unittest
import json
import io
import sys
import os
import numpy as np
from PIL import Image

# ─── Import Flask app ─────────────────────────────────────────────────────────
try:
    from app import app as flask_app
except ImportError as e:
    print(f"❌ app.py import nahi hua: {e}")
    print("   Ensure karein ke app.py aur model saath hain.")
    sys.exit(1)


def make_test_image_bytes(size=(256, 256), color=(120, 120, 120), fmt='JPEG'):
    """In-memory test image bytes banata hai"""
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


class TestHomeRoute(unittest.TestCase):
    """/ route tests"""

    def setUp(self):
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

    def test_home_returns_200(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200,
                         f"Home page 200 nahi diya: {res.status_code}")

    def test_home_returns_html(self):
        res = self.client.get('/')
        ct  = res.content_type
        self.assertIn('html', ct, f"Content-type HTML nahi: {ct}")

    def test_home_contains_title(self):
        res  = self.client.get('/')
        body = res.data.decode('utf-8', errors='ignore')
        self.assertIn('Brain', body, "Page mein 'Brain' text nahi mila")


class TestPredictRoute(unittest.TestCase):
    """/ predict endpoint tests"""

    def setUp(self):
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

    def _post_image(self, img_bytes, filename='test.jpg', content_type='image/jpeg'):
        data = {'file': (io.BytesIO(img_bytes), filename, content_type)}
        return self.client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')

    # ── Happy path ─────────────────────────────────────────────────────────────

    def test_predict_returns_200(self):
        img = make_test_image_bytes()
        res = self._post_image(img)
        self.assertEqual(res.status_code, 200,
                         f"Predict 200 nahi diya: {res.status_code}")

    def test_predict_response_is_json(self):
        img = make_test_image_bytes()
        res = self._post_image(img)
        self.assertEqual(res.content_type, 'application/json',
                         f"Response JSON nahi: {res.content_type}")

    def test_predict_has_required_keys(self):
        img     = make_test_image_bytes()
        res     = self._post_image(img)
        data    = json.loads(res.data)
        required = {'class', 'confidence', 'description',
                    'advice', 'all_probs', 'original', 'annotated'}
        missing  = required - set(data.keys())
        self.assertEqual(missing, set(), f"Response mein keys missing: {missing}")

    def test_predict_class_is_valid(self):
        img    = make_test_image_bytes()
        res    = self._post_image(img)
        data   = json.loads(res.data)
        valid  = {'glioma', 'meningioma', 'notumor', 'pituitary'}
        self.assertIn(data['class'], valid,
                      f"Class invalid: {data['class']}")

    def test_predict_confidence_in_range(self):
        img   = make_test_image_bytes()
        res   = self._post_image(img)
        data  = json.loads(res.data)
        conf  = data['confidence']
        self.assertGreaterEqual(conf, 0.0,  f"Confidence < 0: {conf}")
        self.assertLessEqual(conf,    100.0, f"Confidence > 100: {conf}")

    def test_predict_all_probs_four_classes(self):
        img   = make_test_image_bytes()
        res   = self._post_image(img)
        data  = json.loads(res.data)
        probs = data['all_probs']
        self.assertEqual(len(probs), 4,
                         f"all_probs mein 4 classes honi chahiye: {probs}")

    def test_predict_all_probs_sum_near_100(self):
        img   = make_test_image_bytes()
        res   = self._post_image(img)
        data  = json.loads(res.data)
        total = sum(data['all_probs'].values())
        self.assertAlmostEqual(total, 100.0, delta=1.0,
                               msg=f"Probabilities sum ≠ 100: {total}")

    def test_predict_description_not_empty(self):
        img  = make_test_image_bytes()
        res  = self._post_image(img)
        data = json.loads(res.data)
        self.assertTrue(len(data['description']) > 5,
                        "Description bahut choti ya empty hai")

    def test_predict_advice_not_empty(self):
        img  = make_test_image_bytes()
        res  = self._post_image(img)
        data = json.loads(res.data)
        self.assertTrue(len(data['advice']) > 5,
                        "Advice bahut choti ya empty hai")

    def test_predict_original_image_is_base64(self):
        import base64
        img    = make_test_image_bytes()
        res    = self._post_image(img)
        data   = json.loads(res.data)
        b64str = data['original']
        try:
            decoded = base64.b64decode(b64str)
            self.assertGreater(len(decoded), 100,
                               "Decoded original image bahut choti hai")
        except Exception as e:
            self.fail(f"Original image valid base64 nahi: {e}")

    def test_predict_annotated_image_is_base64(self):
        import base64
        img    = make_test_image_bytes()
        res    = self._post_image(img)
        data   = json.loads(res.data)
        b64str = data['annotated']
        try:
            decoded = base64.b64decode(b64str)
            self.assertGreater(len(decoded), 100,
                               "Decoded annotated image bahut choti hai")
        except Exception as e:
            self.fail(f"Annotated image valid base64 nahi: {e}")

    def test_predict_png_image(self):
        img = make_test_image_bytes(fmt='PNG')
        res = self._post_image(img, filename='test.png', content_type='image/png')
        self.assertEqual(res.status_code, 200)

    def test_predict_large_image(self):
        """Badi image bhi process honi chahiye"""
        img = make_test_image_bytes(size=(1024, 1024))
        res = self._post_image(img)
        self.assertEqual(res.status_code, 200)

    def test_predict_small_image(self):
        """Choti image bhi resize hokar kaam karni chahiye"""
        img = make_test_image_bytes(size=(64, 64))
        res = self._post_image(img)
        self.assertEqual(res.status_code, 200)

    # ── Error handling ────────────────────────────────────────────────────────

    def test_predict_no_file_returns_400(self):
        res = self.client.post('/predict',
                               data={},
                               content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400,
                         "File ke baghair 400 aana chahiye")

    def test_predict_no_file_returns_error_message(self):
        res  = self.client.post('/predict',
                                data={},
                                content_type='multipart/form-data')
        data = json.loads(res.data)
        self.assertIn('error', data, "Error key response mein honi chahiye")

    def test_predict_empty_filename_returns_400(self):
        data = {'file': (io.BytesIO(b''), '', 'image/jpeg')}
        res  = self.client.post('/predict',
                                data=data,
                                content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)

    def test_predict_corrupt_image_returns_500(self):
        """Corrupt bytes pe 500 ya error aana chahiye"""
        bad_bytes = b'this is not an image at all!!'
        res       = self._post_image(bad_bytes, 'bad.jpg')
        self.assertIn(res.status_code, [400, 500],
                      f"Corrupt image pe {res.status_code} aaya")

    def test_invalid_route_returns_404(self):
        res = self.client.get('/nonexistent')
        self.assertEqual(res.status_code, 404)


class TestResponseConsistency(unittest.TestCase):
    """Same image do baar bhejne par same result aana chahiye"""

    def setUp(self):
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

    def test_deterministic_prediction(self):
        img_bytes = make_test_image_bytes()

        def post():
            data = {'file': (io.BytesIO(img_bytes), 'test.jpg', 'image/jpeg')}
            res  = self.client.post('/predict',
                                    data=data,
                                    content_type='multipart/form-data')
            return json.loads(res.data)

        r1 = post()
        r2 = post()
        self.assertEqual(r1['class'],      r2['class'],
                         "Same image par alag class aa rahi hai!")
        self.assertAlmostEqual(r1['confidence'], r2['confidence'], places=2,
                               msg="Same image par confidence alag hai!")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Brain Tumor Flask API — Test Suite")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [TestHomeRoute, TestPredictRoute, TestResponseConsistency]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("✅  Saare API tests pass ho gaye!")
    else:
        print(f"❌  {len(result.failures)} failures, {len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
