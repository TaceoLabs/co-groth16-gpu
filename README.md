# Setup Instructions for ICICLE Backend

This project depends on the **ICICLE** native backend from the  
`icicle-snark` repository.

Because the backend must be compiled with specific flags you must perform the setup manually before 
running the code or tests.

_TODO: build.rs_

---

## Prerequisites

Ensure the following tools are installed and available in your `PATH`:

- **Git**
- **CMake**
- **C/C++ toolchain**
- **CUDA toolkit** (required when using the local CUDA backend)

---

## 1. Clone `icicle-snark`


```bash
git clone https://github.com/ingonyama-zk/icicle-snark.git
```

## 2. Build the ICICLE backend

Navigate to the ICICLE directory and configure CMake with the required flags.

Example for the **bn254** curve using the **local CUDA backend**:

```bash
cd icicle-snark/icicle

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURVE=bn254 \
  -DCUDA_BACKEND=local

cmake --build build -j
```

---

## Changing curves

CMake caches configuration **per build directory**.

If you re-build icicle switching to different curve (for example `bn254 → bls12_377`), you must
remove the cached `FIELD`flag:


```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCURVE=bn254 \
  -DCUDA_BACKEND=local

cmake --build build -j
```

Failing to do this may produce the following error:

```
CURVE and FIELD should not be defined at the same time unless they are equal
```

---

## 3. Export the backend path

After a successful build, set the required environment variable:

```bash
export ICICLE_BACKEND_INSTALL_DIR=path/to/icicle-snark/icicle/build/backend
```

This variable **must be set** before running:

* the application
* tests
* any binaries that depend on ICICLE

---