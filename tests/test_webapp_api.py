import urllib.request
import json

def test_api():
    base = 'http://127.0.0.1:8000'

    # 1. Root HTML
    with urllib.request.urlopen(base) as r:
        html = r.read().decode('utf-8')
        assert 'DRIVER GUARDIAN' in html, 'HTML does not contain title'
        print("[PASS] GET / -> Cockpit HTML loaded successfully")

    # 2. Status
    with urllib.request.urlopen(f"{base}/api/status") as r:
        st = json.loads(r.read())
        assert st["status"] == "online"
        print(f"[PASS] GET /api/status -> status={st['status']}, active_model={st['active_model_name']}")

    # 3. Models
    with urllib.request.urlopen(f"{base}/api/models") as r:
        md = json.loads(r.read())
        assert len(md["models"]) >= 5
        print(f"[PASS] GET /api/models -> {len(md['models'])} models listed")

    # 4. Simulation Frame
    with urllib.request.urlopen(f"{base}/api/simulation/frame") as r:
        sf = json.loads(r.read())
        assert "image" in sf and "metrics" in sf
        print(f"[PASS] GET /api/simulation/frame -> class={sf['metrics']['class_name']}, fatigue={sf['metrics']['fatigue_score']:.2f}")

    # 5. XAI Snapshot
    req = urllib.request.Request(
        f"{base}/api/diagnose_snapshot",
        data=json.dumps({"image": ""}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        xai_res = json.loads(r.read())
        assert xai_res["status"] == "success"
        print(f"[PASS] POST /api/diagnose_snapshot -> XAI attributions calculated")

    # 6. Report Export
    with urllib.request.urlopen(f"{base}/api/export_report") as r:
        rep = json.loads(r.read())
        assert "report_id" in rep
        print(f"[PASS] GET /api/export_report -> report_id={rep['report_id']}")

    print("\n=======================================================")
    print(" ALL 6 WEB APP TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================")

if __name__ == "__main__":
    test_api()
