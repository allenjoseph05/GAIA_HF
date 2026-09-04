"""External service clients."""

from gaia_max.clients.gaia_api import (
    GaiaApiClient,
    GaiaApiDownloadTooLargeError,
    GaiaApiError,
    GaiaApiNotFoundError,
    GaiaApiResponseError,
    GaiaApiTransientError,
    RawAttachment,
)
from gaia_max.clients.hf_gaia_dataset import (
    GaiaGatedDatasetClient,
    GatedDatasetAccessError,
    GatedDatasetAttachmentNotFoundError,
    GatedDatasetDownloadTooLargeError,
    GatedDatasetError,
    GatedDatasetPolicyError,
    GatedDatasetResponseError,
    GatedDatasetTransientError,
    GatedRawAttachment,
)

__all__ = [
    "GaiaApiClient",
    "GaiaApiDownloadTooLargeError",
    "GaiaApiError",
    "GaiaApiNotFoundError",
    "GaiaApiResponseError",
    "GaiaApiTransientError",
    "GaiaGatedDatasetClient",
    "GatedDatasetAccessError",
    "GatedDatasetAttachmentNotFoundError",
    "GatedDatasetDownloadTooLargeError",
    "GatedDatasetError",
    "GatedDatasetPolicyError",
    "GatedDatasetResponseError",
    "GatedDatasetTransientError",
    "GatedRawAttachment",
    "RawAttachment",
]
