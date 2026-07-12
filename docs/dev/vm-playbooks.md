# VM playbooks

Stages 1-4 are implemented by agents working on VMs that must build the tree and run its tests
unattended. So these playbooks are exact, re-runnable commands, not prose. "Install the CUDA
toolkit" is useless to an agent mid-ticket.

**Every step is idempotent**: safe to re-run on a half-provisioned VM. Where that is not obvious,
the step says why. Each playbook ends with a verification command and the output you should see.

| Backend | Stage | Host requirement |
|---|---|---|
| [Linux CPU](#linux-cpu) | 0-1 | Any x86_64 or arm64 Linux VM |
| [macOS Metal](#macos-metal) | 2 | **Real Apple Silicon hardware** or Apple-virtualized macOS |
| [Linux CUDA](#linux-cuda) | 3 | NVIDIA GPU with a passed-through or native driver |
| [Linux Vulkan](#linux-vulkan) | 4 | Any Linux VM (**lavapipe** needs no GPU) |

---

## Linux CPU

The baseline. CPU is a first-class target *and* the correctness oracle for every kernel
(ROADMAP §4), so this playbook is the one that must never break.

**1. Toolchain.** `apt-get install` is idempotent — already-installed packages are a no-op.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip ccache pkg-config
```

**2. Clone with submodules.** Re-running `submodule update` on an initialized tree is a no-op.

```bash
git clone https://github.com/dillon-blake/llama-farm.git ~/llama-farm || true
cd ~/llama-farm
git submodule update --init --recursive
./scripts/apply-patches.sh
```

**3. Virtualenv.** `python3 -m venv` on an existing venv leaves it alone.

```bash
cd ~/llama-farm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install scikit-build-core cmake ninja pytest ruff numpy pre-commit build
```

**4. Build.** Cap the jobs if the VM has under ~2 GB of RAM per core — ggml's CPU kernels are
heavy TUs and swapping makes the build slower, not just tighter.

```bash
cd ~/llama-farm
source .venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install -e . --no-build-isolation
pip install -e vendor/llama.cpp/gguf-py
```

**5. Vendored test binaries.**

```bash
cd ~/llama-farm
source .venv/bin/activate
cmake -S vendor/llama.cpp -B build/vendor-tests \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
cmake --build build/vendor-tests --target test-backend-ops -j$(nproc)
```

**6. Verify.**

```bash
pytest tests/ -m "not slow"
./build/vendor-tests/bin/test-backend-ops test -b CPU -o MUL_MAT,OUT_PROD,SOFT_MAX
```

Expected: pytest reports **all tests passed**, and `test-backend-ops` ends with
`Backend CPU: OK` and a non-zero count of cases run. A run of `0 tests` means your `-o` filter
matched nothing — a typo, not a pass.

---

## macOS Metal

**Metal cannot be virtualized on a Linux hypervisor.** There is no GPU passthrough path. This
playbook needs real Apple Silicon, or macOS virtualized *on* Apple Silicon (Virtualization.
framework, UTM, Tart), which does expose the GPU.

**1. Toolchain.** `xcode-select --install` exits non-zero if already installed; the `|| true`
is what makes the step idempotent.

```bash
xcode-select --install || true
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
brew install cmake ninja ccache python@3.12
```

**2-3. Clone and virtualenv.** Identical to the Linux CPU playbook, steps 2-3.

**4. Build with Metal.** Metal defaults **ON** for Apple builds
(`vendor/llama.cpp/ggml/CMakeLists.txt:239`), so a build that says nothing about Metal gets it.
Being explicit costs nothing and documents intent.

```bash
cd ~/llama-farm && source .venv/bin/activate
CMAKE_ARGS="-DGGML_METAL=ON" pip install -e . --no-build-isolation
pip install -e vendor/llama.cpp/gguf-py

cmake -S vendor/llama.cpp -B build/vendor-tests \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_METAL=ON
cmake --build build/vendor-tests --target test-backend-ops -j$(sysctl -n hw.ncpu)
```

**5. Verify.**

```bash
./build/vendor-tests/bin/test-backend-ops support -b Metal
```

Expected: a `Metal` device is listed, and each op prints `supported` or `not supported`. Until
the stage-2 kernels land, the backward ops (`SOFT_MAX_BACK`, `RMS_NORM_BACK`, `OUT_PROD`, …) will
say **not supported** — that is the correct starting state, and it is exactly what S2-02…S2-09
change. `ggml_backend_sched` falls back to CPU for them, so training still *works*, just not
GPU-resident.

If no Metal device appears, the VM has no GPU access; see the hardware note above.

---

## Linux CUDA

**1. Driver and toolkit.** Pin a known-good range: CUDA **12.4-12.6** with driver ≥ 550. Re-running
the apt install is a no-op.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip ccache

# NVIDIA's repo. Re-running is safe; the keyring package overwrites itself.
distro=$(. /etc/os-release; echo "$ID${VERSION_ID//./}")
wget -qO /tmp/cuda-keyring.deb \
  "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**2. Check the GPU is visible before building anything.**

```bash
nvidia-smi
nvcc --version
```

Expected: `nvidia-smi` prints a table with your GPU and a driver version ≥ 550. If it errors,
stop — a CUDA build against an invisible GPU compiles fine and then fails at runtime, which is a
much more confusing failure.

**3-4. Clone and virtualenv.** Identical to the Linux CPU playbook, steps 2-3.

**5. Build with CUDA.** This is slow — nvcc compiles a lot of template instantiations.

```bash
cd ~/llama-farm && source .venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -e . --no-build-isolation
pip install -e vendor/llama.cpp/gguf-py

cmake -S vendor/llama.cpp -B build/vendor-tests \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_CUDA=ON
cmake --build build/vendor-tests --target test-backend-ops -j$(nproc)
```

Set `CMAKE_CUDA_ARCHITECTURES` to your card's compute capability to cut build time
substantially (e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` for Ampere consumer cards).

**6. Verify.**

```bash
./build/vendor-tests/bin/test-backend-ops support -b CUDA0
```

Expected: a `CUDA0` device is listed with your GPU's name. `OUT_PROD` will report **not
supported** for quantized `src0` until S3-02 lands — that is the one op standing between CUDA and
GPU-resident dense training.

---

## Linux Vulkan

**lavapipe** (Mesa's software Vulkan) means shader work can be correctness-tested on **any Linux
VM, with no GPU at all**. It is a *functional* path, not a performance path — expect it to be
slow, and never benchmark on it.

**1. Vulkan SDK + lavapipe.**

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip ccache \
    libvulkan-dev vulkan-tools glslc spirv-tools mesa-vulkan-drivers
```

`mesa-vulkan-drivers` is what provides lavapipe. `glslc` compiles the shaders; without it the
Vulkan backend configures but produces no pipelines.

**2. Select the lavapipe ICD.** This is the step people miss. Without it, Vulkan finds *no*
device on a GPU-less VM and the backend silently does not register.

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
echo 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json' >> ~/.bashrc
vulkaninfo --summary | head -20
```

Expected: `driverName = llvmpipe` (lavapipe's Vulkan driver) and `deviceType =
PHYSICAL_DEVICE_TYPE_CPU`. On a VM with a *real* GPU, unset `VK_ICD_FILENAMES` to use it instead.

**3-4. Clone and virtualenv.** Identical to the Linux CPU playbook, steps 2-3.

**5. Build with Vulkan.**

```bash
cd ~/llama-farm && source .venv/bin/activate
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install -e . --no-build-isolation
pip install -e vendor/llama.cpp/gguf-py

cmake -S vendor/llama.cpp -B build/vendor-tests \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DGGML_VULKAN=ON
cmake --build build/vendor-tests --target test-backend-ops -j$(nproc)
```

**6. Verify.**

```bash
./build/vendor-tests/bin/test-backend-ops support -b Vulkan0
```

Expected: a `Vulkan0` device is listed — named `llvmpipe` under lavapipe, or your real GPU
otherwise. If nothing is listed, `VK_ICD_FILENAMES` is not set (step 2).

---

## Self-hosted GitHub Actions runners

GitHub-hosted runners cannot provide Metal, CUDA, or a real Vulkan GPU, so the `ci-metal`,
`ci-cuda`, and `ci-vulkan` lanes need self-hosted runners on the VMs above.

Provision the VM with its backend playbook **first** — the runner inherits that environment.

```bash
# On the VM, as the user that owns the build environment.
mkdir -p ~/actions-runner && cd ~/actions-runner

RUNNER_VERSION=2.321.0
curl -o actions-runner.tar.gz -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf ./actions-runner.tar.gz            # re-extracting over an existing runner is safe

# Get a registration token: Settings -> Actions -> Runners -> New self-hosted runner.
# Tokens are single-use and expire in an hour.
./config.sh --url https://github.com/dillon-blake/llama-farm \
            --token <REGISTRATION_TOKEN> \
            --labels self-hosted,linux,cuda \
            --unattended --replace

sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

`--replace` is what makes re-registration idempotent: it takes over the existing runner of the
same name instead of erroring.

**Labels are the contract with the workflows.** Use exactly:

| Backend | Labels |
|---|---|
| Linux CUDA | `self-hosted,linux,cuda` |
| Linux Vulkan (native GPU) | `self-hosted,linux,vulkan` |
| macOS Metal | `self-hosted,macos,metal` |

On macOS use the `osx-arm64` runner tarball and `./svc.sh install` (no `sudo`).

Job names follow the `ci-<backend> / <job>` convention from S0-07, so branch-protection rules
stay uniform across lanes.

**Verify:** the runner shows as **Idle** under Settings → Actions → Runners, and a
`workflow_dispatch` of its lane picks it up.
