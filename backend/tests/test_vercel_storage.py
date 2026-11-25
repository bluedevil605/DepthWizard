import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from vercel import blob

from app import main


def test_blob_path_must_stay_inside_uploads():
    main._validate_blob_pathname("uploads/session-image.tif")

    with pytest.raises(HTTPException) as exc_info:
        main._validate_blob_pathname("uploads/../private.tif")

    assert exc_info.value.status_code == 400


def test_blob_processing_is_disabled_until_enabled(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("BLOB_UPLOADS_ENABLED", "false")
    request = main.BlobProcessRequest(
        input_pathname="uploads/session-image.tif",
        input_filename="image.tif",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.process_blob_upload(request))

    assert exc_info.value.status_code == 503


def test_generated_asset_urls_are_published_to_blob(tmp_path, monkeypatch):
    asset = tmp_path / "mesh.glb"
    asset.write_bytes(b"terrain-mesh")
    monkeypatch.setattr(main, "OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "OUTPUT_URL_CACHE", {})
    monkeypatch.setenv("VERCEL", "1")

    async def missing_blob(_pathname):
        raise blob.BlobNotFoundError()

    async def upload_blob(pathname, body, **_kwargs):
        contents = b"".join([chunk async for chunk in body])
        assert contents == b"terrain-mesh"
        return SimpleNamespace(url=f"https://demo.public.blob.vercel-storage.com/{pathname}")

    monkeypatch.setattr(blob, "head_blob", missing_blob)
    monkeypatch.setattr(blob, "put_blob", upload_blob)

    result = asyncio.run(
        main._publish_output_urls(
            {"glb_url": "/static/outputs/mesh.glb", "preview_url": None}
        )
    )

    assert result == {
        "glb_url": "https://demo.public.blob.vercel-storage.com/outputs/shared/mesh.glb",
        "preview_url": None,
    }
