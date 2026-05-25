def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_ready_removed(client):
    response = client.get("/ready")
    assert response.status_code == 404


def test_admin_removed(client):
    assert client.post("/admin/reload-model").status_code == 404
    assert client.post("/admin/threshold", json={"threshold": 0.7}).status_code == 404


def test_detect_removed(client, sample_image_bytes):
    response = client.post(
        "/detect",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 404


def test_submit_and_status(client, sample_image_bytes):
    submit_response = client.post(
        "/submit",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert submit_response.status_code == 202
    submit_body = submit_response.json()
    assert submit_body["status"] == "processing"
    assert submit_body["task_id"]

    status_response = client.get(f"/status/{submit_body['task_id']}")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["task_id"] == submit_body["task_id"]
    assert status_body["status"] in {"processing", "completed", "failed"}
    if status_body["status"] == "completed":
        assert status_body["result"]["result"]["label"] in {"real", "fake"}


def test_status_unknown_task(client):
    response = client.get("/status/not-a-real-task")
    assert response.status_code == 404
