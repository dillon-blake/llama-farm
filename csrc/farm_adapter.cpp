// Read a LoRA adapter's tensors back out -- the G2 enumeration ABI (S1-08).
//
// Training writes the adapter's A/B tensors in place. To SAVE the result, something has to read
// them back, and llama.cpp offers no way to: `ab_map` lives in a private header
// (src/llama-adapter.h), so the tensors are unreachable from outside the vendored tree. That is
// what this file is for, and it is the same reason farm_train.cpp exists.
//
// It supersedes the ll_debug_* accessors of S1-03, which addressed tensors by name and needed a
// live training context. These need neither: an adapter handle is enough, so saving does not
// require the model to still be in training mode.

#include "farm_api.h"

#include "llama-adapter.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The adapter's base-tensor names, sorted.
//
// ab_map is an unordered_map, so its iteration order is an implementation detail -- it is not
// promised to be the same between two builds, let alone two libraries. An enumeration API whose
// index meant something different on Tuesday would be a fine way to write a corrupt adapter file,
// so the order is imposed here and is the same everywhere: lexicographic by base tensor name.
std::vector<std::string> sorted_names(const llama_adapter_lora * adapter) {
    std::vector<std::string> names;
    names.reserve(adapter->ab_map.size());

    for (const auto & [name, weight] : adapter->ab_map) {
        names.push_back(name);
    }

    std::sort(names.begin(), names.end());

    return names;
}

} // namespace

int32_t ll_adapter_n_tensors(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    return (int32_t)adapter->ab_map.size();
}

int32_t ll_adapter_tensor_info(llama_adapter_lora * adapter, int32_t index, char * name_out, int32_t name_capacity,
                               int64_t * ne_a, int64_t * ne_b) {
    if (adapter == nullptr || name_out == nullptr || ne_a == nullptr || ne_b == nullptr) {
        return LL_ERR_INVALID_ARG;
    }

    const std::vector<std::string> names = sorted_names(adapter);

    if (index < 0 || (size_t)index >= names.size()) {
        return LL_ERR_INVALID_ARG;
    }

    const std::string & name = names[(size_t)index];

    if ((int32_t)name.size() + 1 > name_capacity) {
        return LL_ERR_INVALID_ARG;
    }

    std::memcpy(name_out, name.c_str(), name.size() + 1);

    const llama_adapter_lora_weight & weight = adapter->ab_map.at(name);

    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        ne_a[d] = weight.a->ne[d];
        ne_b[d] = weight.b->ne[d];
    }

    return LL_OK;
}

// out may be NULL with n_max == 0, which asks only "how many elements?" -- the same two-call
// convention llama_tokenize uses, so a caller can size its buffer without guessing.
int64_t ll_adapter_get(llama_adapter_lora * adapter, int32_t index, bool is_b, float * out, int64_t n_max) {
    if (adapter == nullptr || n_max < 0 || (out == nullptr && n_max != 0)) {
        return LL_ERR_INVALID_ARG;
    }

    const std::vector<std::string> names = sorted_names(adapter);

    if (index < 0 || (size_t)index >= names.size()) {
        return LL_ERR_INVALID_ARG;
    }

    const llama_adapter_lora_weight & weight = adapter->ab_map.at(names[(size_t)index]);
    ggml_tensor * tensor = is_b ? weight.b : weight.a;

    if (tensor == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (tensor->type != GGML_TYPE_F32) {
        return LL_ERR_TENSOR_NOT_F32;
    }

    const int64_t n = ggml_nelements(tensor);

    if (out == nullptr) {
        return n; // the sizing call
    }
    if (n > n_max) {
        return LL_ERR_INVALID_ARG;
    }

    // ggml_backend_tensor_get, not a memcpy from tensor->data: an adapter tensor need not live in
    // host memory. Today it always does (stage 1 is CPU-only), and today a memcpy would work --
    // which is exactly why it is worth not writing, because on the first GPU backend it would
    // become a read of a device pointer from the host and the failure would look like corruption.
    ggml_backend_tensor_get(tensor, out, 0, n * sizeof(float));

    return n;
}

int64_t ll_adapter_set(llama_adapter_lora * adapter, int32_t index, bool is_b, const float * data, int64_t n) {
    if (adapter == nullptr || data == nullptr || n < 0) {
        return LL_ERR_INVALID_ARG;
    }

    const std::vector<std::string> names = sorted_names(adapter);

    if (index < 0 || (size_t)index >= names.size()) {
        return LL_ERR_INVALID_ARG;
    }

    const llama_adapter_lora_weight & weight = adapter->ab_map.at(names[(size_t)index]);
    ggml_tensor * tensor = is_b ? weight.b : weight.a;

    if (tensor == nullptr) {
        return LL_ERR_INVALID_ARG;
    }
    if (tensor->type != GGML_TYPE_F32) {
        return LL_ERR_TENSOR_NOT_F32;
    }
    if (ggml_nelements(tensor) != n) {
        return LL_ERR_SHAPE_MISMATCH;
    }

    ggml_backend_tensor_set(tensor, data, 0, n * sizeof(float));

    return n;
}
