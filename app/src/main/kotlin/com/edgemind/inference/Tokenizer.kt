package com.edgemind.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Tokenizer wrapper backed by ExecuTorch's pytorch_tokenizers (auto-detects
 * HF JSON, TikToken, SentencePiece, or BPE format from the loaded file).
 * Falls back to whitespace tokenization for mock mode.
 */
class Tokenizer(private val context: Context, private val tokenizerAsset: String) {

    private val nativeAvailable: Boolean
    private var nativeHandle: Long = 0L

    /** True only when the native library AND a tokenizer file are loaded and ready. */
    val isReady: Boolean get() = nativeAvailable && nativeHandle != 0L

    init {
        var available = false
        try {
            System.loadLibrary("executorch_jni")
            available = true
        } catch (_: UnsatisfiedLinkError) {}
        nativeAvailable = available
    }

    suspend fun load() = withContext(Dispatchers.IO) {
        if (nativeAvailable) {
            try {
                val assetPath = copyAssetToCache(tokenizerAsset)
                nativeHandle = nativeLoadTokenizer(assetPath)
                if (nativeHandle == 0L) {
                    Log.w(TAG, "Native load returned null handle for $tokenizerAsset — mock mode active")
                }
            } catch (e: java.io.FileNotFoundException) {
                Log.w(TAG, "Tokenizer asset not found: $tokenizerAsset — mock mode active")
            }
        }
    }

    suspend fun encode(text: String, maxLength: Int = 2048): LongArray =
        withContext(Dispatchers.Default) {
            if (isReady) {
                nativeEncode(nativeHandle, text, maxLength)
            } else {
                mockEncode(text, maxLength)
            }
        }

    suspend fun decode(tokenIds: LongArray): String =
        withContext(Dispatchers.Default) {
            if (isReady) {
                nativeDecode(nativeHandle, tokenIds)
            } else {
                mockDecode(tokenIds)
            }
        }

    fun close() {
        if (nativeAvailable && nativeHandle != 0L) {
            nativeCloseTokenizer(nativeHandle)
            nativeHandle = 0L
        }
    }

    private fun copyAssetToCache(asset: String): String {
        val outFile = context.cacheDir.resolve(asset)
        if (!outFile.exists()) {
            context.assets.open("models/$asset").use { input ->
                outFile.outputStream().use { input.copyTo(it) }
            }
        }
        return outFile.absolutePath
    }

    private fun mockEncode(text: String, maxLength: Int): LongArray {
        val tokens = text.lowercase().split(Regex("\\s+"))
            .map { it.hashCode().toLong().and(0x7FFF) }
            .take(maxLength)
        return LongArray(tokens.size) { tokens[it] }
    }

    private fun mockDecode(ids: LongArray): String = ids.joinToString(" ") { "<$it>" }

    private external fun nativeLoadTokenizer(path: String): Long
    private external fun nativeEncode(handle: Long, text: String, maxLength: Int): LongArray
    private external fun nativeDecode(handle: Long, tokenIds: LongArray): String
    private external fun nativeCloseTokenizer(handle: Long)

    companion object {
        private const val TAG = "Tokenizer"
    }
}
