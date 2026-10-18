from pathlib import Path

from knowledge.processor.import_process.nodes import node_md_img


def test_every_local_image_gets_vision_description(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.png").write_bytes(b"image-a")
    (image_dir / "b.jpg").write_bytes(b"image-b")
    markdown_path = tmp_path / "demo.md"
    markdown_path.write_text(
        "![A](images/a.png)\n![B](images/b.jpg)\n![C](https://example.com/c.png)",
        encoding="utf-8",
    )
    descriptions = []

    node = node_md_img.NodeMDImg()
    monkeypatch.setattr(node_md_img, "get_minio_client", lambda: object())
    monkeypatch.setattr(
        node,
        "_describe_image",
        lambda path, fallback: descriptions.append(path) or f"视觉描述-{fallback}",
    )
    monkeypatch.setattr(
        node,
        "_step_3_upload_image",
        lambda client, path, state: f"http://minio/{Path(path).name}",
    )

    state = {
        "md_content": markdown_path.read_text(encoding="utf-8"),
        "md_path": str(markdown_path),
        "file_title": "demo",
    }
    node.process(state)

    assert len(descriptions) == 3
    assert "视觉描述-A" in state["md_content"]
    assert "视觉描述-B" in state["md_content"]
    assert "视觉描述-C" in state["md_content"]
