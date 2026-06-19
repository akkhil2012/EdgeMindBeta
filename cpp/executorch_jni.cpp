/**
 * ExecuTorch JNI bridge for EdgeMind.
 *
 * Build requirements:
 *   - ExecuTorch built + installed for arm64-v8a via
 *     executorch/scripts/build_android_library.sh (produces
 *     executorch/cmake-out-android-arm64-v8a, including the
 *     executorch-config.cmake package consumed by app/CMakeLists.txt).
 *
 * When that install is absent, find_package(executorch) fails and the
 * native library still compiles without EXECUTORCH_AVAILABLE defined;
 * ExecuTorchRunner.kt detects the missing symbols and falls back to its
 * Kotlin mock implementation automatically.
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>

#define LOG_TAG "ExecuTorchJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// ── Conditional ExecuTorch include ───────────────────────────────────────
#ifdef EXECUTORCH_AVAILABLE
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/runtime/executor/method_meta.h>
#include <pytorch/tokenizers/tokenizer.h>
using namespace executorch::runtime;
using namespace executorch::aten;
using namespace executorch::extension;
#endif

// ── Model / tokenizer handle registries ─────────────────────────────────────
// Map opaque jlong handles → instance pointers.
// In production use an AtomicLong registry; kept simple here for clarity.

#ifdef EXECUTORCH_AVAILABLE
static std::unordered_map<jlong, std::unique_ptr<Module>> g_modules;
static jlong g_next_handle = 1;

static std::unordered_map<jlong, std::unique_ptr<::tokenizers::Tokenizer>> g_tokenizers;
static jlong g_next_tokenizer_handle = 1;
#endif

extern "C" {

/**
 * Load a .pte model from the given absolute path.
 * Returns a non-zero handle on success, 0 on failure.
 */
JNIEXPORT jlong JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeLoad(
        JNIEnv* env, jobject /* this */, jstring path_jstr) {
#ifdef EXECUTORCH_AVAILABLE
    const char* path = env->GetStringUTFChars(path_jstr, nullptr);
    LOGI("Loading model: %s", path);

    auto module = std::make_unique<Module>(path, Module::LoadMode::MmapUseMlock);
    auto err = module->load();
    env->ReleaseStringUTFChars(path_jstr, path);

    if (err != Error::Ok) {
        LOGE("Module::load failed: %d", (int)err);
        return 0L;
    }

    jlong handle = g_next_handle++;
    g_modules[handle] = std::move(module);
    LOGI("Model loaded, handle=%lld", (long long)handle);

    auto meta = g_modules[handle]->method_meta("forward");
    if (meta.ok()) {
        size_t n = meta->num_inputs();
        LOGI("forward() expects %zu input(s)", n);
        for (size_t i = 0; i < n; i++) {
            auto info = meta->input_tensor_meta(i);
            if (info.ok()) {
                std::string shape;
                for (auto s : info->sizes()) {
                    shape += std::to_string(s) + ",";
                }
                LOGI("  input[%zu]: shape=[%s] dtype=%d", i, shape.c_str(), (int)info->scalar_type());
            } else {
                auto tag = meta->input_tag(i);
                LOGI("  input[%zu]: non-tensor tag=%d", i, tag.ok() ? (int)*tag : -1);
            }
        }
    } else {
        LOGE("method_meta(forward) failed: %d", (int)meta.error());
    }

    return handle;
#else
    LOGW("ExecuTorch not compiled in — nativeLoad returning 0");
    return 0L;
#endif
}

/**
 * Forward pass with LongArray input (token IDs).
 * Returns FloatArray of logits.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeForward(
        JNIEnv* env, jobject /* this */, jlong handle, jlongArray input_ids) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it == g_modules.end()) {
        LOGE("Invalid handle: %lld", (long long)handle);
        return env->NewFloatArray(0);
    }

    jsize len = env->GetArrayLength(input_ids);
    jlong* ids = env->GetLongArrayElements(input_ids, nullptr);

    // Build EValue input tensor [1, seq_len]
    std::vector<int64_t> input_data(ids, ids + len);
    env->ReleaseLongArrayElements(input_ids, ids, JNI_ABORT);

    std::vector<SizesType> sizes{1, static_cast<SizesType>(len)};
    auto tensor = from_blob(input_data.data(), sizes, ScalarType::Long);

    auto result = it->second->forward({EValue(*tensor)});
    if (!result.ok()) {
        LOGE("forward() failed: error=%d", (int)result.error());
        return env->NewFloatArray(0);
    }

    auto out_tensor = result.get()[0].toTensor();
    auto* data = out_tensor.data_ptr<float>();
    jsize out_size = (jsize)out_tensor.numel();

    jfloatArray jout = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(jout, 0, out_size, data);
    return jout;
#else
    return env->NewFloatArray(0);
#endif
}

/**
 * Forward pass with FloatArray input (embeddings / audio features).
 */
JNIEXPORT jfloatArray JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeForwardFloat(
        JNIEnv* env, jobject /* this */,
        jlong handle, jfloatArray input, jlongArray shape_arr) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it == g_modules.end()) {
        return env->NewFloatArray(0);
    }

    jsize in_len = env->GetArrayLength(input);
    jfloat* in_data = env->GetFloatArrayElements(input, nullptr);

    jsize shape_len = env->GetArrayLength(shape_arr);
    jlong* shape_data = env->GetLongArrayElements(shape_arr, nullptr);
    std::vector<SizesType> shape(shape_data, shape_data + shape_len);
    env->ReleaseLongArrayElements(shape_arr, shape_data, JNI_ABORT);

    auto tensor = from_blob(in_data, shape, ScalarType::Float);
    auto result = it->second->forward({EValue(*tensor)});
    env->ReleaseFloatArrayElements(input, in_data, JNI_ABORT);

    if (!result.ok()) {
        return env->NewFloatArray(0);
    }

    auto out_tensor = result.get()[0].toTensor();
    auto* out_data = out_tensor.data_ptr<float>();
    jsize out_size = (jsize)out_tensor.numel();

    jfloatArray jout = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(jout, 0, out_size, out_data);
    return jout;
#else
    return env->NewFloatArray(0);
#endif
}

/**
 * Release the Module and free memory.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_inference_ExecuTorchRunner_nativeClose(
        JNIEnv* /* env */, jobject /* this */, jlong handle) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_modules.find(handle);
    if (it != g_modules.end()) {
        g_modules.erase(it);
        LOGI("Model unloaded, handle=%lld", (long long)handle);
    }
#endif
}

// ── Tokenizer JNI ─────────────────────────────────────────────────────────
// Backed by executorch::extension::llm::load_tokenizer(), which auto-detects
// HF JSON, TikToken, SentencePiece, or BPE format from the file contents.

/**
 * Load a tokenizer model/config file from the given absolute path.
 * Returns a non-zero handle on success, 0 on failure.
 */
JNIEXPORT jlong JNICALL
Java_com_edgemind_inference_Tokenizer_nativeLoadTokenizer(
        JNIEnv* env, jobject /* this */, jstring path_jstr) {
#ifdef EXECUTORCH_AVAILABLE
    const char* path = env->GetStringUTFChars(path_jstr, nullptr);
    LOGI("Loading tokenizer: %s", path);
    auto tokenizer = ::executorch::extension::llm::load_tokenizer(path);
    env->ReleaseStringUTFChars(path_jstr, path);

    if (tokenizer == nullptr) {
        LOGE("load_tokenizer failed");
        return 0L;
    }

    jlong handle = g_next_tokenizer_handle++;
    g_tokenizers[handle] = std::move(tokenizer);
    LOGI("Tokenizer loaded, handle=%lld", (long long)handle);
    return handle;
#else
    LOGW("ExecuTorch not compiled in — nativeLoadTokenizer returning 0");
    return 0L;
#endif
}

JNIEXPORT jlongArray JNICALL
Java_com_edgemind_inference_Tokenizer_nativeEncode(
        JNIEnv* env, jobject /* this */, jlong handle, jstring text, jint max_length) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_tokenizers.find(handle);
    if (it == g_tokenizers.end()) {
        LOGE("Invalid tokenizer handle: %lld", (long long)handle);
        return env->NewLongArray(0);
    }

    const char* text_chars = env->GetStringUTFChars(text, nullptr);
    std::string input(text_chars);
    env->ReleaseStringUTFChars(text, text_chars);

    auto result = it->second->encode(input, /*bos=*/1, /*eos=*/0);
    if (!result.ok()) {
        LOGE("encode() failed");
        return env->NewLongArray(0);
    }

    const auto& ids = result.get();
    jsize n = (jsize)std::min<size_t>(ids.size(), (size_t)max_length);
    std::vector<jlong> out(ids.begin(), ids.begin() + n);

    jlongArray jout = env->NewLongArray(n);
    env->SetLongArrayRegion(jout, 0, n, out.data());
    return jout;
#else
    return env->NewLongArray(0);
#endif
}

JNIEXPORT jstring JNICALL
Java_com_edgemind_inference_Tokenizer_nativeDecode(
        JNIEnv* env, jobject /* this */, jlong handle, jlongArray token_ids) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_tokenizers.find(handle);
    if (it == g_tokenizers.end()) {
        LOGE("Invalid tokenizer handle: %lld", (long long)handle);
        return env->NewStringUTF("");
    }

    jsize len = env->GetArrayLength(token_ids);
    jlong* ids = env->GetLongArrayElements(token_ids, nullptr);

    std::string output;
    uint64_t prev = it->second->bos_tok();
    for (jsize i = 0; i < len; i++) {
        uint64_t cur = (uint64_t)ids[i];
        auto piece = it->second->decode(prev, cur);
        if (piece.ok()) {
            output += *piece;
        }
        prev = cur;
    }
    env->ReleaseLongArrayElements(token_ids, ids, JNI_ABORT);
    return env->NewStringUTF(output.c_str());
#else
    return env->NewStringUTF("");
#endif
}

/**
 * Release the Tokenizer and free memory.
 */
JNIEXPORT void JNICALL
Java_com_edgemind_inference_Tokenizer_nativeCloseTokenizer(
        JNIEnv* /* env */, jobject /* this */, jlong handle) {
#ifdef EXECUTORCH_AVAILABLE
    auto it = g_tokenizers.find(handle);
    if (it != g_tokenizers.end()) {
        g_tokenizers.erase(it);
        LOGI("Tokenizer unloaded, handle=%lld", (long long)handle);
    }
#endif
}

// ── LoRA training JNI ─────────────────────────────────────────────────────

JNIEXPORT jfloat JNICALL
Java_com_edgemind_training_LoraTrainer_nativeTrainLora(
        JNIEnv* env, jobject /* this */,
        jstring model_asset, jstring adapter_output_path,
        jobjectArray samples, jint batch_size,
        jfloat learning_rate, jint max_steps,
        jint lora_rank, jint lora_alpha, jint fragment_size) {
    // Placeholder — real impl calls ExecuTorch training API
    LOGW("nativeTrainLora: ExecuTorch training not compiled in");
    return 0.0f;
}

} // extern "C"
