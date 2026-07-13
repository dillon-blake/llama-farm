"""S0-04: the ctypes binding layer loads, resolves every symbol, and locks the vendor commit."""

import ctypes
import math
import pkgutil
import shutil
import subprocess
import sys

import pytest

from learning_llamas import _ffi
from learning_llamas._ffi import ggml_opt, loader, registry

from .vendor_pin import vendored_commit


@pytest.fixture(scope="module")
def libs() -> loader.Libraries:
    return _ffi.load()


def test_load_order_is_the_documented_contract() -> None:
    """libggml-base → libggml → libllama → liblearningllamas (csrc/README.md).

    Each library depends on the ones before it, and RTLD_GLOBAL puts their symbols in the
    global namespace so that GGML_BACKEND_DL backends dlopen'd later can resolve ggml_*
    against the already-loaded libggml-base.
    """
    assert [lib.value for lib in loader.LOAD_ORDER] == [
        "ggml-base",
        "ggml",
        "llama",
        "learningllamas",
    ]


def test_all_four_libraries_open(libs: loader.Libraries) -> None:
    for handle in (libs.ggml_base, libs.ggml, libs.llama, libs.farm):
        assert isinstance(handle, ctypes.CDLL)


def test_every_declared_symbol_resolves(libs: loader.Libraries) -> None:
    """The tripwire for vendor drift beyond the commit lock: a renamed function fails here."""
    handles = {
        registry.Library.GGML_BASE: libs.ggml_base,
        registry.Library.GGML: libs.ggml,
        registry.Library.LLAMA: libs.llama,
        registry.Library.FARM: libs.farm,
    }
    for symbol in _ffi.SYMBOLS:
        assert getattr(handles[symbol.library], symbol.name) is not None


@pytest.mark.skipif(
    shutil.which("nm") is None, reason="needs binutils' nm to read the symbol table"
)
@pytest.mark.skipif(sys.platform != "linux", reason="ELF-specific: reads .so dynamic symbol tables")
def test_every_symbol_is_declared_in_the_library_that_DEFINES_it() -> None:
    """Declaring a symbol against the wrong library is silent, and then it segfaults.

    The test above cannot catch this, and that is not a gap in it — it is how dlopen works.
    ``libggml`` links ``libggml-base``, so ``getattr(libs.ggml, "ggml_scale")`` *succeeds*: the
    dynamic linker walks the dependency chain and finds it. ctypes then hangs the declared
    ``argtypes``/``restype`` on a function object obtained from the wrong handle, the call goes out
    through an unconfigured path, and a returned 64-bit pointer comes back truncated to 32 bits.
    What you see is a segfault, several calls later, somewhere else entirely.

    This has now cost two debugging sessions — once on a base symbol read through ``libs.ggml``,
    once on ``ggml_backend_cpu_buffer_from_ptr`` in S1-13. So it gets a test: a symbol must be
    declared against the library whose own symbol table *defines* it, not merely one that can
    resolve it.
    """
    lib_dir = loader.library_dir()

    defined: dict[registry.Library, set[str]] = {}
    for library in registry.Library:
        path = lib_dir / loader.library_filename(library.value)
        out = subprocess.run(  # noqa: S603
            ["nm", "-D", "--defined-only", str(path)],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        defined[library] = {line.split()[-1] for line in out.splitlines() if line.strip()}

    misplaced = []
    for symbol in _ffi.SYMBOLS:
        if symbol.name in defined[symbol.library]:
            continue
        actual = [lib.value for lib, names in defined.items() if symbol.name in names]
        misplaced.append(f"{symbol.name}: declared in {symbol.library.value}, defined in {actual}")

    assert not misplaced, "symbols declared against a library that does not define them:\n  " + (
        "\n  ".join(misplaced)
    )


@pytest.mark.parametrize(
    "name",
    [
        "llama_set_adapters_lora",
        "llama_opt_init",
        "llama_opt_epoch",
        "ggml_opt_prepare_alloc",
        "ggml_opt_get_constant_optimizer_params",
        "ggml_backend_tensor_set",
        "ggml_backend_tensor_get",
    ],
)
def test_load_bearing_symbols_are_declared(name: str) -> None:
    """Symbols later tickets depend on must stay in the table.

    llama_set_adapters_lora in particular is the batch attach API present at the pinned commit;
    a vendor bump that drops it must fail loudly rather than silently.
    """
    assert any(symbol.name == name for symbol in _ffi.SYMBOLS), f"{name} is not in the table"


def test_probe_matches_the_version_lock(libs: loader.Libraries) -> None:
    """All three parties to the version lock agree: submodule, native build, Python package."""
    assert libs.farm.ll_probe().decode() == vendored_commit()
    assert loader.VENDORED_COMMIT == vendored_commit()


def test_tampered_commit_lock_raises(libs: loader.Libraries) -> None:
    """A mismatched vendor commit must refuse to run, naming both hashes.

    This is the whole safety story for the hand-written struct mirrors: drift corrupts memory
    rather than failing to link, so the only defense is to not start.
    """
    with pytest.raises(RuntimeError, match="commit mismatch") as excinfo:
        _ffi.load(expected_commit="0000000000000000000000000000000000000000")

    message = str(excinfo.value)
    assert vendored_commit() in message
    assert "0000000000000000000000000000000000000000" in message


def test_default_optimizer_params_round_trip(libs: loader.Libraries) -> None:
    """Call a C function that returns ggml_opt_optimizer_params BY VALUE.

    This is a partial struct-layout check: if the mirror's field order or sizes were wrong, the
    values would come back as garbage rather than plausible AdamW defaults. It is also the
    safe direction of the struct-by-value ABI — see test_no_struct_returning_callback_type.
    """
    params = libs.ggml_base.ggml_opt_get_default_optimizer_params(None)

    assert params.adamw.alpha > 0.0
    for value in (
        params.adamw.alpha,
        params.adamw.beta1,
        params.adamw.beta2,
        params.adamw.eps,
        params.adamw.wd,
        params.sgd.alpha,
        params.sgd.wd,
    ):
        assert math.isfinite(value)

    assert 0.0 < params.adamw.beta1 < 1.0
    assert 0.0 < params.adamw.beta2 < 1.0


def test_model_default_params_round_trip(libs: loader.Libraries) -> None:
    """llama_model_params crosses the ABI by value; a bad mirror shows up as junk defaults.

    Values are llama_model_default_params() at the pinned commit (src/llama-model.cpp:2301).
    """
    params = libs.llama.llama_model_default_params()

    assert params.n_gpu_layers == -1  # "all layers"; not 0
    assert params.split_mode == _ffi.SplitMode.LAYER
    assert params.main_gpu == 0
    assert not params.devices
    assert params.vocab_only is False
    assert params.use_mmap is True
    assert params.use_direct_io is False
    assert params.check_tensors is False
    assert params.use_extra_bufts is True
    assert params.no_alloc is False


def test_context_default_params_round_trip(libs: loader.Libraries) -> None:
    """llama_context_params is the largest by-value struct and the easiest to get wrong.

    Field misalignment does not raise — it silently reads the neighbouring field. So this pins
    every *distinctive* default from llama_context_default_params() (src/llama-context.cpp):
    the exact context sizes, the negative sentinels, the F16 cache types, and the trailing bool
    block. Each one lands on the right field only if every preceding field has the right size.
    """
    params = libs.llama.llama_context_default_params()

    assert params.n_ctx == 512
    assert params.n_batch == 2048
    assert params.n_ubatch == 512
    assert params.n_seq_max == 1
    assert params.n_rs_seq == 0
    assert params.n_outputs_max == 0

    assert params.ctx_type == _ffi.ContextType.DEFAULT
    assert params.attention_type == _ffi.AttentionType.UNSPECIFIED  # -1
    assert params.flash_attn_type == _ffi.FlashAttnType.AUTO  # -1

    assert params.rope_freq_base == 0.0  # "0 = from model"
    assert params.rope_freq_scale == 0.0
    assert params.yarn_ext_factor == -1.0
    assert params.defrag_thold == -1.0

    assert params.type_k == _ffi.GGMLType.F16
    assert params.type_v == _ffi.GGMLType.F16

    # The trailing bool block: misalignment here reads pointer bytes as bools.
    assert params.embeddings is False
    assert params.offload_kqv is True
    assert params.no_perf is True
    assert params.op_offload is True
    assert params.swa_full is True
    assert params.kv_unified is False

    assert params.n_samplers == 0
    assert not params.ctx_other


def test_no_struct_returning_callback_type() -> None:
    """_ffi must define no ctypes callback type returning ggml_opt_optimizer_params by value.

    Struct-by-value returns through a ctypes *callback* are a known ABI trap (BLUEPRINT §3).
    The policy is to never register one: OptimizerParams instead hands ggml the exported
    ggml_opt_get_constant_optimizer_params with a Python-owned struct as its userdata, which
    is also how learning-rate schedules work for free.
    """
    offenders = []
    for module_info in pkgutil.iter_modules(_ffi.__path__):
        module = __import__(f"learning_llamas._ffi.{module_info.name}", fromlist=["_"])
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            restype = getattr(attr, "_restype_", None)
            if restype is ggml_opt.ggml_opt_optimizer_params and issubclass(
                attr,
                ctypes._CFuncPtr,  # type: ignore[attr-defined]
            ):
                offenders.append(f"{module_info.name}.{attr_name}")

    assert not offenders, f"struct-returning ctypes callback types found: {offenders}"


def test_optimizer_params_helper_wires_the_constant_callback(libs: loader.Libraries) -> None:
    """OptimizerParams hands over the exported C function, not a Python callback."""
    params = ggml_opt.OptimizerParams(libs)
    params.adamw.alpha = 1e-4

    get_opt_pars, userdata = params.callback()

    expected = ctypes.cast(
        libs.ggml_base.ggml_opt_get_constant_optimizer_params, ctypes.c_void_p
    ).value
    assert get_opt_pars == expected

    # The C function casts userdata back to the struct: the mutation must survive the round
    # trip, which is exactly the mechanism an LR schedule relies on.
    returned = libs.ggml_base.ggml_opt_get_constant_optimizer_params(userdata)
    assert returned.adamw.alpha == pytest.approx(1e-4)


def test_the_opt_params_mirror_matches_the_C_layout() -> None:
    """``ggml_opt_params`` crosses the ABI by value and holds FUNCTION POINTERS.

    A mirror that disagrees about the layout does not read a wrong number — it calls a wrong
    address. And the disagreement is silent: ctypes will happily build a struct of the wrong
    shape and pass it.

    S1-10 inserted ``grad_clip`` into the middle of this struct (between ``opt_period`` and
    ``get_opt_pars``) and the mirror did not declare it at all. That was harmless *by accident*:
    a float at offset 44 lands in the padding already sitting between an int32 at 40 and an
    8-aligned pointer at 48, so every later field still aligned and ``sizeof`` still came to 72.
    The next field added anywhere before ``get_opt_pars`` would have shifted the callback pointer
    by four bytes.

    These numbers are pinned again in ``csrc/farm_internals.cpp`` with ``static_assert`` /
    ``offsetof``, against the real C header. Break one side and the other complains.
    """
    params = ggml_opt.ggml_opt_params

    assert ctypes.sizeof(params) == 72

    expected = {
        "backend_sched": 0,
        "ctx_compute": 8,
        "inputs": 16,
        "outputs": 24,
        "loss_type": 32,
        "build_type": 36,
        "opt_period": 40,
        "grad_clip": 44,
        "get_opt_pars": 48,
        "get_opt_pars_ud": 56,
        "optimizer": 64,
    }

    actual = {name: getattr(params, name).offset for name, _ in params._fields_}
    assert actual == expected
