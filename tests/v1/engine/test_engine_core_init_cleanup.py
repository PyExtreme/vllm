# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def engine_setup(monkeypatch: pytest.MonkeyPatch):
    class FakeExecutor:
        instances = []
        cleanup_error = None

        def __init__(self, config):
            self.shutdown_calls = 0
            self.instances.append(self)

        def shutdown(self):
            self.shutdown_calls += 1
            if self.cleanup_error is not None:
                raise self.cleanup_error

    class FakeScheduler:
        instances = []
        init_error = None

        def __init__(self, **kwargs):
            if self.init_error is not None:
                raise self.init_error
            self.shutdown_calls = 0
            self.instances.append(self)

        def shutdown(self):
            self.shutdown_calls += 1

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=True),
        scheduler_config=SimpleNamespace(
            get_scheduler_cls=lambda: FakeScheduler,
            enable_chunked_prefill=False,
        ),
    )
    monkeypatch.setattr(core_module.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", False)
    monkeypatch.setattr(core_module, "StructuredOutputManager", lambda config: object())
    monkeypatch.setattr(
        core_module, "resolve_kv_cache_block_sizes", lambda *args: (16, 16)
    )
    monkeypatch.setattr(
        EngineCore,
        "_initialize_kv_caches",
        lambda self, config: SimpleNamespace(kv_cache_groups=[object()]),
    )
    return config, FakeExecutor, FakeScheduler


def test_executor_shutdown_when_kv_initialization_fails(engine_setup, monkeypatch):
    config, executor_class, scheduler_class = engine_setup
    original_error = RuntimeError("original init failure")

    def fail_kv_initialization(self, config):
        raise original_error

    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", fail_kv_initialization)

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    assert executor_class.instances[0].shutdown_calls == 1
    assert scheduler_class.instances == []


def test_executor_shutdown_when_scheduler_construction_fails(engine_setup):
    config, executor_class, scheduler_class = engine_setup
    original_error = RuntimeError("scheduler init failure")
    scheduler_class.init_error = original_error

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    assert executor_class.instances[0].shutdown_calls == 1
    assert scheduler_class.instances == []


def test_executor_and_scheduler_shutdown_after_scheduler_construction(
    engine_setup, monkeypatch
):
    config, executor_class, scheduler_class = engine_setup
    original_error = RuntimeError("original init failure")

    def fail_after_scheduler(self):
        raise original_error

    monkeypatch.setattr(
        EngineCore, "_initialize_effective_attention_block_size", fail_after_scheduler
    )

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    assert executor_class.instances[0].shutdown_calls == 1
    assert scheduler_class.instances[0].shutdown_calls == 1


def test_scheduler_shutdown_and_original_error_survive_executor_cleanup_failure(
    engine_setup, monkeypatch
):
    config, executor_class, scheduler_class = engine_setup
    original_error = RuntimeError("original init failure")
    executor_class.cleanup_error = RuntimeError("executor cleanup failure")

    def fail_after_scheduler(self):
        raise original_error

    monkeypatch.setattr(
        EngineCore, "_initialize_effective_attention_block_size", fail_after_scheduler
    )

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    assert executor_class.instances[0].shutdown_calls == 1
    assert scheduler_class.instances[0].shutdown_calls == 1


def test_gc_unfreeze_after_failure_following_successful_freeze(
    engine_setup, monkeypatch
):
    config, executor_class, scheduler_class = engine_setup
    original_error = RuntimeError("original init failure")
    freeze = Mock()
    unfreeze = Mock()

    config.speculative_config = None
    config.model_config = SimpleNamespace(is_diffusion=False, runner_type="generate")
    config.max_concurrent_batches = 1
    config.ec_transfer_config = None
    config.cache_config = SimpleNamespace(enable_prefix_caching=False)
    config.scheduler_config.async_scheduling = False
    monkeypatch.setattr(scheduler_class, "connector", None, raising=False)
    monkeypatch.setattr(scheduler_class, "ec_connector", None, raising=False)
    monkeypatch.setattr(
        scheduler_class, "get_kv_connector", lambda self: None, raising=False
    )
    monkeypatch.setattr(
        core_module,
        "MULTIMODAL_REGISTRY",
        SimpleNamespace(engine_receiver_cache_from_config=lambda config: None),
    )
    monkeypatch.setattr(core_module, "freeze_gc_heap", freeze)
    monkeypatch.setattr(core_module.gc, "unfreeze", unfreeze)
    monkeypatch.setattr(
        core_module, "maybe_attach_gc_debug_callback", Mock(side_effect=original_error)
    )

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    freeze.assert_called_once_with()
    unfreeze.assert_called_once_with()
    assert executor_class.instances[0].shutdown_calls == 1
    assert scheduler_class.instances[0].shutdown_calls == 1


def test_gc_unfreeze_not_called_when_initialization_fails_before_freeze(
    engine_setup, monkeypatch
):
    config, executor_class, _ = engine_setup
    original_error = RuntimeError("original init failure")
    freeze = Mock()
    unfreeze = Mock()

    def fail_kv_initialization(self, config):
        raise original_error

    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", fail_kv_initialization)
    monkeypatch.setattr(core_module, "freeze_gc_heap", freeze)
    monkeypatch.setattr(core_module.gc, "unfreeze", unfreeze)

    with pytest.raises(RuntimeError) as exc_info:
        EngineCore(config, executor_class, log_stats=False)

    assert exc_info.value is original_error
    freeze.assert_not_called()
    unfreeze.assert_not_called()
    assert executor_class.instances[0].shutdown_calls == 1
