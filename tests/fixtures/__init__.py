"""The fixture-model generators, plus the one thing their cache keys have to share.

Every generator caches its GGUF under a directory named by its ``cache_key()``, a hash of the
generator's own source -- which is what makes ``tests/conftest.py``'s promise true, that editing a
generator cannot leave a stale model behind. Three of the four are not self-contained though:
:mod:`gen_tiny_moe`, :mod:`gen_tiny_mamba` and :mod:`gen_tiny_mamba2` import ``CHAT_TEMPLATE`` and
``_load_vocab`` (and the MoE also ``_quantize``/``_QUANT_TYPE``/``_FILE_TYPE``) from
:mod:`gen_tiny_llama`, so their fixture *bytes* depend on a file their key did not hash. Change the
chat template and the llama fixtures regenerate while a warm ``tests/.fixtures`` goes on serving MoE
and Mamba GGUFs carrying the old one, silently, for as long as the cache lives. :func:`llama_source`
closes that: the three siblings fold it into their payload.

It is deliberately the WHOLE llama generator rather than the handful of symbols they import.
Hashing ``inspect.getsource(_load_vocab)`` would miss ``REFERENCE_VOCAB_GGUF``, the constant that
decides *which* vocab those tokens are sliced from; over-hashing costs an unnecessary regeneration
of three tiny models, under-hashing costs a wrong fixture that nothing reports.

:mod:`gen_tiny_llama` itself does not use this helper and must not start using it. Its ``cache_key``
is baked into the convergence reference-curve identity (``tests/convergence/reference_curve.json``,
``reference_curve_wd.json``), so *any* edit to that file -- adding an import included -- forces both
curves to be re-recorded against torch/peft. That is also why its ``build(seed=...)`` argument is
still missing from its key while the three siblings now fold ``seed`` in: nothing calls it with a
non-default seed today, and the fix has to wait for a change that re-records the curves anyway.
"""

from __future__ import annotations

import pathlib


def llama_source() -> str:
    """:mod:`gen_tiny_llama`'s source, read as text with newlines normalized to LF.

    Normalization is not cosmetic: git hands a Windows checkout a CRLF working tree, so hashing the
    raw bytes gives the same generator a different key -- and therefore a different fixture
    directory -- per platform. The long version is in
    :func:`tests.fixtures.gen_tiny_llama.cache_key`, which learned it the hard way on ci-windows.
    """
    from . import gen_tiny_llama

    return pathlib.Path(gen_tiny_llama.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
