import logging
import os

import cloudinary
import cloudinary.uploader
import requests

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloudManager")


class CloudManager:
    def __init__(self):
        if settings.CLOUDINARY_CLOUD_NAME:
            cloudinary.config(
                cloud_name=settings.CLOUDINARY_CLOUD_NAME,
                api_key=settings.CLOUDINARY_API_KEY,
                api_secret=settings.CLOUDINARY_API_SECRET,
                secure=True,
            )

    def _upload_file(self, local_path, cloud_name):
        """Helper internal untuk upload file"""
        if not os.path.exists(local_path):
            return
        try:
            cloudinary.uploader.upload(
                local_path,
                resource_type="raw",
                public_id=cloud_name,
                overwrite=True,
                invalidate=True,
            )
            logger.info(f"☁️ Uploaded: {cloud_name}")
        except Exception as e:
            logger.error(f"Upload Failed {cloud_name}: {e}")

    def _download_file(self, local_path, cloud_name):
        """Helper internal untuk download file"""
        try:
            url = cloudinary.utils.cloudinary_url(cloud_name, resource_type="raw")[0]
            r = requests.get(url)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(r.content)
                logger.info(f"☁️ Downloaded: {cloud_name}")
                return True
        except:
            pass
        return False

    def upload_model(self):
        # UPLOAD SATU PAKET (Model + Scaler)
        self._upload_file(settings.MODEL_FILE, settings.CLOUD_MODEL_NAME)
        # Beri nama unik untuk scaler di cloud
        self._upload_file(settings.SCALER_FILE, "forex_ai_scaler.pkl")

    def download_model(self):
        # DOWNLOAD SATU PAKET
        m = self._download_file(settings.MODEL_FILE, settings.CLOUD_MODEL_NAME)
        s = self._download_file(settings.SCALER_FILE, "forex_ai_scaler.pkl")

        if m and s:
            logger.info("✅ Model & Scaler synced from Cloud.")
            return True
        return False


cloud_manager = CloudManager()
