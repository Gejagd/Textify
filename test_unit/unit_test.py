import sys
import os
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_processor import ImageProcessor
from image_exporter import ImageExporter
from image_source import ImageSourceFactory, FileImageSource, CameraImageSource
from document_detector import DocumentDetector
from finger_remover import FingerRemover
from app import TextifyApp


def _make_bgr(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:30, 10:30] = [0, 0, 0]
    img[0:10, 0:10] = [255, 255, 255]
    return img


def _make_gray(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.full((h, w), 200, dtype=np.uint8)
    img[20:40, 20:40] = 50
    return img


def _make_image_with_document(h: int = 200, w: int = 200) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[30:170, 30:170] = [240, 240, 240]
    img[50:60, 50:150] = [0, 0, 0]
    return img


class TestDocumentDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DocumentDetector()

    def test_order_points_shape_and_order(self):
        pts = np.array([[100, 100], [0, 100], [100, 0], [0, 0]], dtype=np.float32)
        ordered = self.detector.order_points(pts)
        self.assertEqual(ordered.shape, (4, 2))
        np.testing.assert_array_equal(ordered[0], [0, 0])
        np.testing.assert_array_equal(ordered[2], [100, 100])

    def test_find_document_contour_success(self):
        img = _make_image_with_document()
        contour = self.detector.find_document_contour(img)
        self.assertIsNotNone(contour)
        self.assertEqual(contour.shape, (4, 2))

    def test_find_document_contour_none_when_blank(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        contour = self.detector.find_document_contour(blank)
        self.assertIsNone(contour)

    def test_warp_perspective_shape(self):
        img = _make_image_with_document()
        pts = np.array([[30, 30], [170, 30], [170, 170], [30, 170]], dtype=np.float32)
        warped = self.detector.warp_perspective(img, pts)
        self.assertGreater(warped.shape[0], 0)
        self.assertGreater(warped.shape[1], 0)
        self.assertEqual(len(warped.shape), 3)


class TestFingerRemover(unittest.TestCase):
    def setUp(self):
        self.remover = FingerRemover()

    def test_detect_skin_mask_shape(self):
        img = _make_bgr(100, 100)
        mask = self.remover.detect_skin_mask(img)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)

    def test_detect_skin_mask_grayscale_input_returns_zeros(self):
        gray = _make_gray(50, 50)
        mask = self.remover.detect_skin_mask(gray)
        self.assertEqual(mask.shape, (50, 50))
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_remove_fingers_leaves_clean_document_intact(self):
        img = np.full((80, 80, 3), 255, dtype=np.uint8)
        cleaned = self.remover.remove_fingers(img)
        self.assertEqual(cleaned.shape, img.shape)

    def test_remove_fingers_on_skin_patch(self):
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        skin_bgr = [140, 170, 220]
        img[40:60, 40:60] = skin_bgr
        cleaned = self.remover.remove_fingers(img)
        self.assertEqual(cleaned.shape, img.shape)


class TestImageProcessorGrayscale(unittest.TestCase):
    def setUp(self):
        self.proc = ImageProcessor()

    def test_bgr_converts_to_grayscale(self):
        bgr = _make_bgr()
        gray = self.proc.to_grayscale(bgr)
        self.assertEqual(len(gray.shape), 2)

    def test_already_gray_unchanged_shape(self):
        gray = _make_gray()
        result = self.proc.to_grayscale(gray)
        self.assertEqual(result.shape, gray.shape)
        self.assertEqual(len(result.shape), 2)

    def test_output_dtype_uint8(self):
        bgr = _make_bgr()
        gray = self.proc.to_grayscale(bgr)
        self.assertEqual(gray.dtype, np.uint8)


class TestImageProcessorDenoise(unittest.TestCase):
    def setUp(self):
        self.proc = ImageProcessor()

    def test_output_same_shape(self):
        gray = _make_gray()
        blurred = self.proc.denoise(gray)
        self.assertEqual(blurred.shape, gray.shape)

    def test_output_dtype_uint8(self):
        gray = _make_gray()
        blurred = self.proc.denoise(gray)
        self.assertEqual(blurred.dtype, np.uint8)

    def test_blurring_reduces_variance(self):
        noisy = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        blurred = self.proc.denoise(noisy)
        self.assertLess(float(np.var(blurred)), float(np.var(noisy)))


class TestImageProcessorBinarize(unittest.TestCase):
    def setUp(self):
        self.proc = ImageProcessor()

    def test_output_only_0_and_255(self):
        gray = _make_gray()
        blurred = self.proc.denoise(gray)
        binary = self.proc.binarize(blurred)
        unique = np.unique(binary)
        for v in unique:
            self.assertIn(v, [0, 255])

    def test_output_same_shape(self):
        gray = _make_gray()
        blurred = self.proc.denoise(gray)
        binary = self.proc.binarize(blurred)
        self.assertEqual(binary.shape, gray.shape)

    def test_output_dtype_uint8(self):
        gray = _make_gray()
        blurred = self.proc.denoise(gray)
        binary = self.proc.binarize(blurred)
        self.assertEqual(binary.dtype, np.uint8)


class TestImageProcessorCleanNoise(unittest.TestCase):
    def setUp(self):
        self.proc = ImageProcessor()

    def test_output_same_shape(self):
        binary = np.full((64, 64), 255, dtype=np.uint8)
        binary[32, 32] = 0
        cleaned = self.proc.clean_noise(binary)
        self.assertEqual(cleaned.shape, binary.shape)

    def test_isolated_noise_pixel_removed(self):
        binary = np.full((64, 64), 255, dtype=np.uint8)
        binary[20:50, 20:50] = 0
        binary[35, 35] = 255
        cleaned = self.proc.clean_noise(binary)
        self.assertEqual(int(cleaned[35, 35]), 0)

    def test_large_text_region_preserved(self):
        binary = np.full((64, 64), 255, dtype=np.uint8)
        binary[20:50, 20:50] = 0
        cleaned = self.proc.clean_noise(binary)
        center_val = int(cleaned[35, 35])
        self.assertEqual(center_val, 0)


class TestImageProcessorFullPipeline(unittest.TestCase):
    def setUp(self):
        self.proc = ImageProcessor()

    def test_process_bgr_returns_2d(self):
        bgr = _make_bgr()
        result = self.proc.process(bgr)
        self.assertEqual(len(result.shape), 2)

    def test_process_result_binary(self):
        bgr = _make_bgr()
        result = self.proc.process(bgr)
        unique = np.unique(result)
        for v in unique:
            self.assertIn(v, [0, 255])

    def test_process_does_not_mutate_input(self):
        bgr = _make_bgr()
        original = bgr.copy()
        self.proc.process(bgr)
        np.testing.assert_array_equal(bgr, original)

    def test_process_crops_detected_document(self):
        img = _make_image_with_document()
        result = self.proc.process(img)
        self.assertEqual(len(result.shape), 2)
        self.assertLess(result.shape[0], img.shape[0])


class TestImageExporter(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        self.exporter = ImageExporter(output_dir=Path(self.tmp_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_creates_file(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        path = self.exporter.export(img, "test_out")
        self.assertTrue(path.exists())

    def test_export_adds_png_extension(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        path = self.exporter.export(img, "no_ext")
        self.assertEqual(path.suffix, ".png")

    def test_export_preserves_png_extension(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        path = self.exporter.export(img, "already.png", auto_increment=False)
        self.assertEqual(path.name, "already.png")

    def test_export_returns_correct_path(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        path = self.exporter.export(img, "myfile")
        self.assertIn("myfile", str(path))

    def test_export_auto_increments_filenames(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        path1 = self.exporter.export(img, "doc")
        path2 = self.exporter.export(img, "doc")
        path3 = self.exporter.export(img, "doc")
        self.assertEqual(path1.name, "doc_01.png")
        self.assertEqual(path2.name, "doc_02.png")
        self.assertEqual(path3.name, "doc_03.png")

    def test_export_creates_dir_if_missing(self):
        new_dir = Path(self.tmp_dir) / "subdir" / "nested"
        exporter = ImageExporter(output_dir=new_dir)
        img = np.full((32, 32), 255, dtype=np.uint8)
        exporter.export(img, "nested_test")
        self.assertTrue(new_dir.exists())

    def test_get_output_dir(self):
        self.assertEqual(self.exporter.get_output_dir(), Path(self.tmp_dir))


class TestImageSourceFactory(unittest.TestCase):
    def test_file_mode_returns_file_source(self):
        source = ImageSourceFactory.create(ImageSourceFactory.MODE_FILE)
        self.assertIsInstance(source, FileImageSource)

    def test_camera_mode_returns_camera_source(self):
        source = ImageSourceFactory.create(ImageSourceFactory.MODE_CAMERA)
        self.assertIsInstance(source, CameraImageSource)

    def test_unknown_mode_defaults_to_file(self):
        source = ImageSourceFactory.create("unknown_mode")
        self.assertIsInstance(source, FileImageSource)


class TestFileImageSource(unittest.TestCase):
    @patch("image_source.cv2.imread")
    @patch("image_source.filedialog.askopenfilename", return_value="/fake/image.png")
    @patch("image_source.tk.Tk")
    def test_acquire_returns_image_on_valid_path(self, mock_tk, mock_dialog, mock_imread):
        mock_tk.return_value = MagicMock()
        mock_imread.return_value = _make_bgr()
        source = FileImageSource()
        result = source.acquire()
        self.assertIsNotNone(result)

    @patch("image_source.filedialog.askopenfilename", return_value="")
    @patch("image_source.tk.Tk")
    def test_acquire_returns_none_on_cancel(self, mock_tk, mock_dialog):
        mock_tk.return_value = MagicMock()
        source = FileImageSource()
        result = source.acquire()
        self.assertIsNone(result)


class TestCameraImageSource(unittest.TestCase):
    @patch("image_source.cv2.VideoCapture")
    def test_acquire_returns_none_when_no_camera(self, mock_cap_class):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap
        source = CameraImageSource()
        result = source.acquire()
        self.assertIsNone(result)

    @patch("image_source.cv2.destroyAllWindows")
    @patch("image_source.cv2.imshow")
    @patch("image_source.cv2.waitKey", side_effect=[27])
    @patch("image_source.cv2.VideoCapture")
    def test_acquire_returns_none_on_esc(self, mock_cap_class, mock_waitkey, mock_imshow, mock_destroy):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _make_bgr())
        mock_cap_class.return_value = mock_cap
        source = CameraImageSource()
        result = source.acquire()
        self.assertIsNone(result)

    @patch("image_source.cv2.destroyAllWindows")
    @patch("image_source.cv2.imshow")
    @patch("image_source.cv2.waitKey", side_effect=[13])
    @patch("image_source.cv2.VideoCapture")
    def test_acquire_returns_frame_on_enter(self, mock_cap_class, mock_waitkey, mock_imshow, mock_destroy):
        frame = _make_bgr()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap_class.return_value = mock_cap
        source = CameraImageSource()
        result = source.acquire()
        self.assertIsNotNone(result)


class TestTextifyApp(unittest.TestCase):
    def _make_app_with_tmp(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        app = TextifyApp()
        app._exporter = ImageExporter(output_dir=Path(tmp))
        return app, tmp

    @patch("builtins.input", return_value="3")
    def test_invalid_choice_returns_error_code(self, mock_input):
        app = TextifyApp()
        code = app.run()
        self.assertEqual(code, 1)

    @patch("image_source.cv2.imread", return_value=None)
    @patch("image_source.filedialog.askopenfilename", return_value="/fake/path.png")
    @patch("image_source.tk.Tk")
    @patch("builtins.input", return_value="1")
    def test_file_choice_no_image_returns_error(self, mock_input, mock_tk, mock_dialog, mock_imread):
        mock_tk.return_value = MagicMock()
        app, tmp = self._make_app_with_tmp()
        code = app.run()
        self.assertEqual(code, 1)

    @patch("image_source.filedialog.askopenfilename", return_value="")
    @patch("image_source.tk.Tk")
    @patch("builtins.input", return_value="1")
    def test_file_choice_cancel_returns_error(self, mock_input, mock_tk, mock_dialog):
        mock_tk.return_value = MagicMock()
        app, tmp = self._make_app_with_tmp()
        code = app.run()
        self.assertEqual(code, 1)

    @patch("image_source.cv2.imread", return_value=_make_bgr())
    @patch("image_source.filedialog.askopenfilename", return_value="/fake/valid.png")
    @patch("image_source.tk.Tk")
    @patch("builtins.input", return_value="1")
    def test_full_flow_file_returns_success(self, mock_input, mock_tk, mock_dialog, mock_imread):
        mock_tk.return_value = MagicMock()
        app, tmp = self._make_app_with_tmp()
        import shutil
        try:
            code = app.run()
            self.assertEqual(code, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
