# Idle exit of server.py, without a GPU: torch / accelerate / omegaconf and the pipeline are stubbed.
#   pip install pytest httpx fastapi uvicorn && python -m pytest tests/test_server_idle.py

import asyncio
import os
import socket
import subprocess
import sys
import textwrap
import time
import types
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUBS = textwrap.dedent(
    """
    import sys, types
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: None)
    sys.modules.setdefault("torch", torch)
    accelerate = types.ModuleType("accelerate"); accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils.set_seed = lambda seed: None
    sys.modules.setdefault("accelerate", accelerate); sys.modules.setdefault("accelerate.utils", accelerate_utils)
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(load=lambda path: types.SimpleNamespace())
    sys.modules.setdefault("omegaconf", omegaconf)
    inference = types.ModuleType("scripts.inference")
    inference.build_pipeline = lambda config, settings: (object(), None)
    sys.modules["scripts.inference"] = inference
    """
)

exec(STUBS)
sys.path.insert(0, REPO_ROOT)
import server  # noqa: E402


def test_zero_never_exits():
    assert not server.should_idle_exit(0, last_used=0, now=10**9, busy=False)
    assert not server.should_idle_exit(-1, last_used=0, now=10**9, busy=False)


def test_busy_never_exits():
    assert not server.should_idle_exit(10, last_used=0, now=10**9, busy=True)


def test_boundary():
    assert not server.should_idle_exit(10, last_used=1000, now=1000 + 599.9, busy=False)
    assert server.should_idle_exit(10, last_used=1000, now=1000 + 600, busy=False)


def test_health_does_not_count_but_other_requests_do(tmp_path):
    from fastapi.testclient import TestClient

    os.makedirs(tmp_path / "references")
    server.settings.data_dir = str(tmp_path)
    server.settings.version = "1.5"
    server.settings.idle_exit_min = 10
    client = TestClient(server.app)  # no `with`: lifespan (model load) does not run
    server.state["last_used"] = 0.0
    assert client.get("/health").status_code == 200
    assert server.state["last_used"] == 0.0
    assert client.get("/reference").status_code == 200
    assert server.state["last_used"] > 0.0
    server.state.clear()


def _watch(idle_exit_min, srv, hold_lock=False):
    server.settings.idle_exit_min = idle_exit_min
    server.state.clear()
    if srv is not None:
        server.state["server"] = srv
    server.touch()
    if hold_lock:
        server.gpu_lock.acquire()

    async def run():
        try:
            await asyncio.wait_for(server.idle_watch(), timeout=1.0)
            return True
        except asyncio.TimeoutError:
            return False

    try:
        return asyncio.run(run())
    finally:
        if server.gpu_lock.locked():
            server.gpu_lock.release()
        server.state.clear()


@pytest.fixture
def fast_checks(monkeypatch):
    monkeypatch.setattr(server, "IDLE_CHECK_SEC", 0.05)


def test_watch_stops_server_and_keeps_the_lock(fast_checks):
    srv = types.SimpleNamespace(should_exit=False)
    server.settings.idle_exit_min = 0.001
    server.state.clear()
    server.state["server"] = srv
    server.touch()

    async def run():
        await asyncio.wait_for(server.idle_watch(), timeout=1.0)
        return server.gpu_lock.locked()

    try:
        assert asyncio.run(run()) is True  # held so no job starts during shutdown
        assert srv.should_exit is True
    finally:
        server.gpu_lock.release()
        server.state.clear()


def test_watch_waits_while_a_job_runs(fast_checks):
    srv = types.SimpleNamespace(should_exit=False)
    assert _watch(0.001, srv, hold_lock=True) is False
    assert srv.should_exit is False


def test_watch_without_uvicorn_server_stays_up(fast_checks):
    assert _watch(0.001, None) is True  # returns, but nothing was stopped
    assert not server.gpu_lock.locked()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_uvicorn_exits_with_code_0(tmp_path):
    """The supervisor promise: idle exit ends the whole process with 0, so "restart on failure" leaves it down."""
    port = _free_port()
    script = STUBS + textwrap.dedent(
        f"""
        sys.path.insert(0, {REPO_ROOT!r})
        import server
        server.IDLE_CHECK_SEC = 0.2
        sys.argv = ["server", "--port", "{port}", "--data_dir", {str(tmp_path)!r}]
        server.main()
        """
    )
    env = dict(os.environ, LIPSYNC_IDLE_EXIT_MIN="0.02")  # 1.2 s
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        health = None
        deadline = time.time() + 10
        while time.time() < deadline and health is None:
            try:
                health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            except OSError:
                time.sleep(0.1)
        assert health and b'"idle_exit_min":0.02' in health
        while proc.poll() is None and time.time() < deadline + 15:  # keep polling /health: it must not keep the server up
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            except OSError:
                pass
            time.sleep(0.1)
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        out = proc.stdout.read().decode(errors="replace")
    assert "shutting down to free the GPU" in out


def test_default_never_exits(tmp_path):
    port = _free_port()
    script = STUBS + textwrap.dedent(
        f"""
        sys.path.insert(0, {REPO_ROOT!r})
        import server
        server.IDLE_CHECK_SEC = 0.1
        sys.argv = ["server", "--port", "{port}", "--data_dir", {str(tmp_path)!r}]
        server.main()
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "LIPSYNC_IDLE_EXIT_MIN"}
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.5)
        assert proc.poll() is None  # still up
        health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
        assert b'"idle_exit_min":0.0' in health
    finally:
        proc.kill()
        proc.wait()
