import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import tempfile
import os
import sys
import time
import unittest
from unittest.mock import Mock, patch

loader = importlib.machinery.SourceFileLoader("xiancun", str(Path(__file__).parents[1] / "xiancun"))
spec = importlib.util.spec_from_loader(loader.name, loader)
xi = importlib.util.module_from_spec(spec)
loader.exec_module(xi)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = patch.object(xi, "STATE", Path(self.directory.name))
        self.state.start()
        self.addCleanup(self.state.stop)
        self.clock = patch.object(xi.time, "monotonic", return_value=1000)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.probe = patch.object(xi, "snapshot")
        self.snapshot = self.probe.start()
        self.addCleanup(self.probe.stop)
        self.terminate = patch.object(xi, "terminate_workers")
        self.terminate.start()
        self.addCleanup(self.terminate.stop)
        self.gpus = [{"index": i, "uuid": f"GPU-{i}", "util": 0, "free": 32000} for i in range(3)]
        self.snapshot.return_value = (self.gpus, [])
        self.monitor = xi.Monitor(dict(xi.DEFAULTS, gpu_count=2, memory_gib=5.0))
        self.pid = 90000
        self.monitor.start_gpu = Mock(side_effect=self.fake_start)

    def fake_start(self, gpu, now):
        self.pid += 1
        index = str(gpu["index"])
        self.monitor.workers[index] = {"pid": self.pid, "started_monotonic": now}
        child = Mock()
        child.poll.return_value = None
        self.monitor.children[index] = child
        xi.atomic_json(xi.STATE / f"heartbeat-{index}.json", {"pid": self.pid, "monotonic": now})

    def ready(self):
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "confirming_idle")
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")
        self.assertEqual(set(self.monitor.workers), {"0", "1"})

    def test_training_yields_then_resumes_without_restart(self):
        self.ready()
        self.snapshot.return_value = (self.gpus, [{"uuid": f"GPU-{i}", "pid": 123 + i} for i in range(3)])
        for gpu in self.gpus:
            gpu["util"] = 80
        with patch.object(xi, "process_utilization", return_value={(i, 123 + i): 80 for i in range(3)}):
            self.monitor.step()
        self.assertEqual(self.monitor.mode, "waiting_training")
        self.assertFalse(self.monitor.workers)
        # 训练停止计算但保留 CUDA 上下文，必须仍能恢复。
        for gpu in self.gpus:
            gpu["util"] = 0
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")
        self.assertEqual(len(self.monitor.workers), 2)

    def test_own_compute_not_misidentified_as_training(self):
        self.ready()
        self.gpus[0]["util"] = self.gpus[1]["util"] = 100
        self.snapshot.return_value = (self.gpus, [{"uuid": "GPU-0", "pid": w["pid"]}
                                                 for w in self.monitor.workers.values()])
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")

    def test_foreign_cuda_context_at_zero_utilization_allows_start(self):
        self.snapshot.return_value = (self.gpus, [{"uuid": "GPU-0", "pid": 123}])
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")
        self.assertEqual(len(self.monitor.workers), 2)

    def shared_context(self):
        self.ready()
        self.gpus[0]["util"] = 98
        self.snapshot.return_value = (self.gpus, [{"uuid": "GPU-0", "pid": 123}])

    def test_idle_context_sharing_busy_keepalive_gpu_does_not_stop_workers(self):
        self.shared_context()
        with patch.object(xi, "process_utilization", return_value={(0, 123): 0}):
            self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")
        self.assertEqual(len(self.monitor.workers), 2)

    def test_training_on_keepalive_gpu_yields_using_process_utilization(self):
        self.shared_context()
        with patch.object(xi, "process_utilization", return_value={(0, 123): 1}):
            self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1", "2"})
        self.assertIn(0, self.monitor.busy_gpus)

    def test_missing_process_metrics_fallback_to_actual_gpu_load(self):
        self.shared_context()
        unmasked = [dict(g, util=0) for g in self.gpus]
        with patch.object(xi, "process_utilization", return_value={(0, 123): None}), \
                patch.object(xi, "unmasked_snapshot", return_value=(unmasked, [])) as sample:
            self.monitor.step()
            sample.assert_called_once()
        self.assertEqual(self.monitor.mode, "computing")

    def test_process_probe_failure_fallback_detects_active_training(self):
        self.shared_context()
        with patch.object(xi, "process_utilization", side_effect=xi.subprocess.TimeoutExpired("pmon", 10)), \
                patch.object(xi, "unmasked_snapshot", return_value=(self.gpus, [])):
            self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1", "2"})
        self.assertIn(0, self.monitor.busy_gpus)

    def test_busy_other_gpu_does_not_block_idle_gpus(self):
        self.gpus[2]["util"] = 100
        self.ready()
        self.assertEqual(set(self.monitor.workers), {"0", "1"})
        self.assertIn(2, self.monitor.busy_gpus)

    def test_each_gpu_has_independent_idle_confirmation(self):
        self.gpus[0]["util"] = self.gpus[2]["util"] = 80
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1"})
        self.gpus[0]["util"] = 0
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1"})
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"0", "1"})

    def test_fallback_always_resumes_only_own_workers_after_query_failure(self):
        workers = {"0": {"pid": 123}, "1": {"pid": 456}}
        self.snapshot.side_effect = [(self.gpus, []), ValueError("N/A")]
        with patch.object(xi, "is_our_worker", side_effect=lambda r: r["pid"] == 123), \
                patch.object(xi.os, "kill") as kill, patch.object(xi.time, "sleep"):
            with self.assertRaises(ValueError):
                xi.unmasked_snapshot(workers)
        self.assertEqual(kill.call_args_list, [unittest.mock.call(123, xi.signal.SIGSTOP),
                                               unittest.mock.call(123, xi.signal.SIGCONT)])

    def test_fallback_measures_one_card_at_a_time_while_other_keeps_computing(self):
        workers = {"0": {"pid": 123}, "1": {"pid": 456}}
        self.snapshot.side_effect = [
            ([dict(g, util=98) for g in self.gpus], []),
            ([dict(g, util=0 if g["index"] == 0 else 98) for g in self.gpus], []),
            ([dict(g, util=0 if g["index"] == 1 else 98) for g in self.gpus], []),
        ]
        with patch.object(xi, "is_our_worker", return_value=True), \
                patch.object(xi.os, "kill") as kill, patch.object(xi.time, "sleep"):
            gpus, _ = xi.unmasked_snapshot(workers)
        self.assertEqual([g["util"] for g in gpus], [0, 0, 98])
        self.assertEqual(kill.call_args_list, [
            unittest.mock.call(123, xi.signal.SIGSTOP), unittest.mock.call(123, xi.signal.SIGCONT),
            unittest.mock.call(456, xi.signal.SIGSTOP), unittest.mock.call(456, xi.signal.SIGCONT),
        ])

    def test_process_parser_preserves_zero_and_unavailable_metrics(self):
        result = Mock(stdout="# gpu pid type sm mem command\n0 123 C 0 0 python\n1 456 C - - python\n2 - - - - -\n")
        with patch.object(xi.subprocess, "run", return_value=result):
            self.assertEqual(xi.process_utilization(), {(0, 123): 0, (1, 456): None})

    def test_probe_failure_retries_and_does_not_stop_work(self):
        self.ready()
        self.snapshot.side_effect = xi.subprocess.TimeoutExpired("nvidia-smi", 10)
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "probe_error")
        self.assertEqual(len(self.monitor.workers), 2)
        self.snapshot.side_effect = None
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")

    def test_failed_probe_resets_idle_confirmation(self):
        self.monitor.step()
        self.snapshot.side_effect = ValueError("N/A")
        self.now.return_value += 30
        self.monitor.step()
        self.snapshot.side_effect = None
        self.monitor.step()
        self.assertFalse(self.monitor.workers)
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(len(self.monitor.workers), 2)

    def test_dead_worker_replaced_after_retry_delay(self):
        self.ready()
        old_pid = self.monitor.workers["0"]["pid"]
        self.monitor.children["0"].poll.return_value = 1
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1"})
        self.now.return_value += 30
        self.monitor.step()
        self.assertNotEqual(self.monitor.workers["0"]["pid"], old_pid)

    def test_hung_compute_replaced_even_if_pid_alive(self):
        self.ready()
        self.now.return_value += 121
        self.monitor.step()
        self.assertFalse(self.monitor.workers)
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(len(self.monitor.workers), 2)

    def test_rotation_keeps_one_card_computing(self):
        self.monitor.config["rotation_hours"] = 0.01
        self.ready()
        keep_pid = self.monitor.workers["1"]["pid"]
        self.now.return_value += 37
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"1", "2"})
        self.assertEqual(self.monitor.workers["1"]["pid"], keep_pid)

    def test_single_gpu_supported(self):
        self.snapshot.return_value = (self.gpus[:1], [])
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"0"})

    def test_low_memory_retries_when_memory_available(self):
        for gpu in self.gpus:
            gpu["free"] = 100
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "retrying")
        self.gpus[2]["free"] = 32000
        self.monitor.step()
        self.assertEqual(set(self.monitor.workers), {"2"})

    def test_pause_expires_and_resumes(self):
        self.ready()
        xi.atomic_json(xi.STATE / "control.json", {"pause_until": xi.time.time() + 60})
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "paused")
        self.assertFalse(self.monitor.workers)
        xi.atomic_json(xi.STATE / "control.json", {"pause_until": 0})
        self.monitor.step()
        self.now.return_value += 30
        self.monitor.step()
        self.assertEqual(self.monitor.mode, "computing")

    def test_pid_reuse_cannot_signal_unrelated_process(self):
        record = {"pid": xi.os.getpid(), "identity": "not-the-real-start-time"}
        self.assertFalse(xi.is_our_worker(record))
        record["identity"] = xi.process_identity(record["pid"])
        self.assertFalse(xi.is_our_worker(record))

    def test_invalid_interval_and_nan_rejected(self):
        for args in (argparse.Namespace(interval=601), argparse.Namespace(memory_gib=float("nan"))):
            with self.assertRaises(ValueError):
                xi.config_from_args(args)

    def test_missing_stored_python_is_replaced_automatically(self):
        xi.atomic_json(xi.STATE / "config.json", {"python": "/does/not/exist"})
        with patch.object(xi, "python_supports_torch", return_value=True):
            config = xi.config_from_args(argparse.Namespace())
        self.assertEqual(config["python"], xi.sys.executable)

    def test_missing_supervisor_uses_builtin_backend(self):
        (xi.STATE / "enabled").touch()
        with patch.object(xi, "supervisor_available", return_value=False), \
                patch.object(xi, "start_local_monitor") as start:
            self.assertEqual(xi.ensure_service(), "builtin")
        self.assertEqual((xi.STATE / "backend").read_text().strip(), "builtin")
        start.assert_called_once_with()


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        state = patch.object(xi, "STATE", Path(directory.name))
        state.start()
        self.addCleanup(state.stop)
        env = patch.dict(os.environ, {"XIANCUN_BACKEND": "auto"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("XIANCUN_PYTHON", "CUDA_VISIBLE_DEVICES", "XIANCUN_TORCH_INDEX_URL"):
            os.environ.pop(key, None)

    def test_isolated_devices_diagnosis_does_not_claim_driver_is_broken(self):
        with patch.object(Path, "glob", return_value=[]), patch.object(Path, "exists", return_value=True):
            result = xi.gpu_diagnosis(RuntimeError("driver query failed"))
        self.assertIn("沙箱隔离", result)
        self.assertIn("--gpus all", result)
        self.assertIn("不是需要重装驱动", result)

    def test_gpu_failure_aborts_start_before_install_or_enabling(self):
        with patch.object(sys, "argv", [str(xi.SELF), "start"]), \
                patch.object(xi, "snapshot", side_effect=RuntimeError("no GPU")), \
                patch.object(xi, "install_runtime") as install:
            with self.assertRaises(RuntimeError):
                xi.main()
        install.assert_not_called()
        self.assertFalse((xi.STATE / "enabled").exists())

    def test_cpu_only_python_is_rejected(self):
        # This exercises the actual subprocess probe with a CPU-only torch
        # stand-in; a successful import must not count as CUDA readiness.
        (xi.STATE / "torch.py").write_text("class cuda:\n @staticmethod\n def is_available(): return False\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(xi.STATE)}):
            self.assertFalse(xi.python_supports_torch(sys.executable))

    def test_no_install_fails_without_downloading(self):
        with patch.object(xi, "python_supports_torch", return_value=False), \
                patch.object(xi, "install_runtime") as install:
            with self.assertRaises(ValueError):
                xi.config_from_args(argparse.Namespace(no_install=True))
        install.assert_not_called()

    def test_missing_torch_bootstraps_private_runtime(self):
        with patch.object(xi, "python_supports_torch", return_value=False), \
                patch.object(xi, "install_runtime", return_value="/private/runtime/bin/python") as install:
            config = xi.config_from_args(argparse.Namespace())
        self.assertEqual(config["python"], "/private/runtime/bin/python")
        install.assert_called_once_with()

    def test_dependency_install_targets_private_venv(self):
        with patch.object(xi, "smi_rows", return_value=[["580.159.04"]]), \
                patch.object(xi.venv, "EnvBuilder") as builder, \
                patch.object(xi.subprocess, "run") as run, \
                patch.object(xi, "python_supports_torch", return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            python = xi.install_runtime()
        self.assertEqual(python, str(xi.STATE / "runtime/bin/python"))
        builder.return_value.create.assert_called_once_with(xi.STATE / "runtime")
        self.assertEqual(run.call_args.args[0][0], python)
        self.assertIn("https://download.pytorch.org/whl/cu128", run.call_args.args[0])

    def test_snapshot_respects_cuda_visible_devices(self):
        rows = [["0", "GPU-zero", "0", "32000"], ["1", "GPU-one", "0", "32000"]]
        for mask in ("1", "GPU-one"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), \
                    patch.object(xi, "smi_rows", side_effect=[rows, []]):
                gpus, _ = xi.snapshot()
            self.assertEqual([g["index"] for g in gpus], [1])

    def test_missing_identity_never_matches_dead_pid(self):
        xi.atomic_json(xi.STATE / "monitor.pid", {"pid": 987654321, "identity": None})
        with patch.object(xi, "process_identity", return_value=None):
            self.assertIsNone(xi.monitor_record())

    @unittest.skipUnless(xi.process_identity(os.getpid()), "需要同一 PID 命名空间下可读的 /proc")
    def test_background_start_idempotency_and_stop_without_supervisor(self):
        # A busy GPU fixture avoids GPU/PyTorch use while running the actual
        # detached monitor and single-instance handshake across processes.
        smi = xi.STATE / "fake-smi"
        smi.write_text("#!/bin/sh\ncase \"$1\" in\n--query-gpu=*) echo '0, GPU-test, 100, 32000';;\nesac\n")
        smi.chmod(0o755)
        xi.atomic_json(xi.STATE / "config.json", dict(xi.DEFAULTS, interval=1, idle_seconds=1))
        (xi.STATE / "enabled").touch()
        with patch.dict(os.environ, {"XIANCUN_NVIDIA_SMI": str(smi), "XIANCUN_BACKEND": "builtin"}):
            try:
                xi.ensure_service()
                first = xi.monitor_record()
                self.assertIsNotNone(first)
                xi.ensure_service()
                self.assertEqual(xi.monitor_record()["pid"], first["pid"])
                deadline = time.monotonic() + 5
                while not (xi.STATE / "status.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(xi.read_json(xi.STATE / "status.json")["mode"], "waiting_training")
            finally:
                xi.stop_service()
                (xi.STATE / "enabled").unlink(missing_ok=True)
        self.assertIsNone(xi.monitor_record())
        self.assertEqual(xi.read_json(xi.STATE / "status.json")["mode"], "stopped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
