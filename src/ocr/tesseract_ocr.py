# tesseract_ocr.py
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


class TesseractOCR:

    def read_field(self, image: np.ndarray, field: str) -> str:
        """
        Read text from a cropped field image
        Tries multiple methods and returns best result
        """
        results = []

        # Try all preprocessing methods
        for method in [1, 2, 3, 4]:
            try:
                processed = self._preprocess(image, method)
                text = self._ocr(processed, field)
                cleaned = self._clean(text, field)
                if cleaned:
                    results.append(cleaned)
            except Exception:
                continue

        if not results:
            return ""

        # For PAN number return the one matching PAN pattern
        if field == "pan_number":
            for r in results:
                if re.match(r"[A-Z]{5}[0-9]{4}[A-Z]", r):
                    return r

        # For DOB return the one with date pattern
        if field == "dob":
            for r in results:
                if re.search(r"\d{2}/\d{2}/\d{4}", r):
                    return r

        # For names return longest result
        return max(results, key=len)

    def _preprocess(self, image: np.ndarray, method: int) -> np.ndarray:
        """
        4 different preprocessing methods
        Method 1 — simple upscale + grayscale
        Method 2 — upscale + otsu threshold
        Method 3 — upscale + sharpen + threshold
        Method 4 — large upscale + denoise + threshold
        """

        # Convert to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]

        if method == 1:
            # Simple — just upscale and grayscale
            scale = max(3, 150 // max(h, 1))
            resized = cv2.resize(
                image,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return gray

        elif method == 2:
            # Upscale + Otsu threshold
            scale = max(3, 150 // max(h, 1))
            resized = cv2.resize(
                image,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh

        elif method == 3:
            # Upscale + sharpen + threshold
            scale = max(4, 200 // max(h, 1))
            resized = cv2.resize(
                image,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Sharpen
            sharpen_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])
            sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

            _, thresh = cv2.threshold(
                sharpened, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh

        elif method == 4:
            # Large upscale + denoise + adaptive threshold
            scale = max(5, 250 // max(h, 1))
            resized = cv2.resize(
                image,
                (w * scale, h * scale),
                interpolation=cv2.INTER_LANCZOS4
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, h=10)

            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11, C=4
            )
            return thresh

        return image

    def _ocr(self, image: np.ndarray, field: str) -> str:
        """Run Tesseract OCR with field specific config"""

        pil_img = Image.fromarray(image)

        # Use different PSM modes for each field
        psm_modes = ["--psm 7", "--psm 6", "--psm 8"]
        best = ""

        for psm in psm_modes:
            try:
                if field == "pan_number":
                    config = f"{psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                elif field == "dob":
                    config = f"{psm} --oem 3 -c tessedit_char_whitelist=0123456789/-"
                else:
                    config = f"{psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"
                text = pytesseract.image_to_string(
                    pil_img,
                    lang="eng",
                    config=config
                )

                if len(text.strip()) > len(best):
                    best = text.strip()

            except Exception:
                continue

        return best

    def _clean(self, text: str, field: str) -> str:
        """Clean OCR output based on field type"""
        if field == "pan_number":
            return self._clean_pan(text)
        elif field == "dob":
            return self._clean_dob(text)
        else:
            return self._clean_name(text)

    @staticmethod
    def _clean_pan(text: str) -> str:
        """Clean PAN number — format AAAAA9999A"""
        text = text.upper().replace(" ", "").replace("\n", "")

        # Find exact PAN pattern
        match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)
        if match:
            return match.group(0)

        # Fix common OCR mistakes
        # O mistaken as 0 in letter positions
        # 0 mistaken as O in number positions
        fixed = ""
        for i, ch in enumerate(text[:10]):
            if i < 5 or i == 9:
                # Letter position — replace 0 with O
                fixed += ch.replace("0", "O").replace("1", "I")
            else:
                # Number position — replace O with 0
                fixed += ch.replace("O", "0").replace("I", "1")

        match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", fixed)
        if match:
            return match.group(0)

        return text[:10]

    @staticmethod
    def _clean_dob(text: str) -> str:
        """Clean date of birth — format DD/MM/YYYY"""
        text = text.replace(" ", "").replace("\n", "")

        # Full date DD/MM/YYYY
        match = re.search(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})", text)
        if match:
            return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"

        # Short year DD/MM/YY
        match = re.search(r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{2})", text)
        if match:
            year = match.group(3)
            full_year = "19" + year if int(year) > 24 else "20" + year
            return f"{match.group(1)}/{match.group(2)}/{full_year}"

        return text

    
    # @staticmethod
    # def _clean_name(text: str) -> str:
    #     """Clean name field"""

    # # Remove non letter characters
    #     text = re.sub(r"[^A-Za-z\s]", "", text)

    # # Known garbage words that appear on PAN cards
    #     garbage_words = [
    #         "tea", "are", "at", "fathers", "name", "father",
    #         "income", "tax", "dept", "government", "india",
    #         "permanent", "account", "number", "card", "col",
    #         "yt", "ee", "ea", "be", "oad", "ast", "lam", "ry",
    #         "og", "peg", "eet", "lake", "pln", "ren", "lg",
    #         "sw", "rx", "ih", "rs", "jo", "wa", "an", "fog",
    #         "oo", "ls", "spouse", "date", "birth", "signature",
    #         "dept", "gov", "of", "the", "to", "in", "my", "a",
    #         "g", "s", "o", "m", "i", "see", "term", "pd",
    #         "ghia", "berg", "ev", "cib", "das", "now", "uf",
    #         "lun", "pada", "ns", "ss", "woe", "im", "af", "pa",
    #         "k", "so", "Piegula", "Sw", "Rx", "piegula"
    #     ]

    #     words = text.split()

    # # Remove garbage words
    #     clean_words = [
    #         w for w in words
    #         if w.lower() not in garbage_words
    #         and len(w) > 2
    #     ]

    # # Real names are max 3 words
    #     if len(clean_words) > 3:
    #     # Take last 3 words — usually the actual name
    #             clean_words = clean_words[-3:]

    #     return " ".join(clean_words).title()


    @staticmethod
    def _clean_name(text: str) -> str:
        """Clean name — keep only letters"""
    # Remove non letter characters
        text = re.sub(r"[^A-Za-z\s]", "", text)
    # Remove extra spaces
        text = " ".join(text.split())
    # Title case
        return text.title()