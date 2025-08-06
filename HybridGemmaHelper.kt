package com.example.sightspeak_offline.hybrid

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.example.sightspeak_offline.mediapipe.AudioAsr
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.genai.llminference.*
import kotlinx.coroutines.*
import java.util.LinkedList

/**
 * HybridGemmaHelper wraps both LLM and vision/image capabilities,
 * plus audio transcription, providing async and sync methods.
 */
class HybridGemmaHelper(
    context: Context,
    private val modelPath: String,   // Path to the .task model
    private val userLang: String     // User’s chosen language for responses
) {
    // Core TFLite LLM inference object
    private val llmInference: LlmInference
    // A session to run queries (text + optional image)
    private  lateinit var llmSession: LlmInferenceSession
    // Offline ASR helper
    private val asr = AudioAsr(context)
    // Job handle for any ongoing generation
    private var generationJob: Job? = null




    // System prompt prefix to shape every conversation
    private val SYSTEM_PREFIX = """
    You are SightSpeak, a multimodal AI assistant for blind users. 
    Describe scenes clearly, practical and simply.Say what objects are presents and where they are. Avoid abstract, poetic, or artistic language. 
    List visible objects, people, and text. Do not ask questions. 
    Avoid repeating similar details. Prioritize useful interpretation over raw visual description.
    When translating, output only the translated result. Do not include any explanation or commentary.
""".trimIndent()


    /** Language‐specific “reply in…” prefix for the model */
    private fun localePrefix(): String = when (userLang) {
        "中文"   -> "请用简体中文回答下面的问题："
        "日本語"  -> "以下の質問には日本語で回答してください："
        "한국어"  -> "다음 질문에 한국어로 대답하세요："
        "Deutsch" -> "Bitte beantworten Sie die folgende Frage auf Deutsch："
        "Español" -> "Por favor responda lo siguiente en español："
        "Français"-> "Veuillez répondre à ce qui suit en français："
        else      -> "Please answer the following in English:"
    }

    // Simple RAG memory buffer of the last N Q&A pairs
    private val memoryLimit = 5
    private val conversationMemory = LinkedList<Pair<String, String>>()

    // Concurrency flags
    @Volatile private var isGenerating = false
    @Volatile private var isBusy = false

    init {
        // 1) Build inference options & create LlmInference
        val llmOptions = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTopK(32)
            .setMaxNumImages(1)
            .build()
        llmInference = LlmInference.createFromOptions(context, llmOptions)

        // 2) Create an initial session with vision enabled
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setGraphOptions(
                GraphOptions.builder().setEnableVisionModality(true).build()
            ).build()
        llmSession = LlmInferenceSession.createFromOptions(llmInference, sessionOptions)
    }

    /**
     * Fully resets the session and clears conversation memory.
     * Safely closes the old session, recreates it, and empties RAG.
     */
    fun resetSession() {
        Log.w("Gemma", "🔁 Clearing conversation memory (session reused)")

        if (isGenerating) {
            Log.w("Gemma", "⚠️ Generation in progress — cannot reset now.")
            return
        }

        generationJob?.cancel()

        // Safe close (avoids invalid JNI ref crash)
        try {
            if (::llmSession.isInitialized) {
                llmSession.close()
            }
        } catch (e: Exception) {
            Log.w("Gemma", "session.close() threw, but continuing", e)
        }

        try {
            val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
                .setGraphOptions(
                    GraphOptions.builder().setEnableVisionModality(true).build()
                ).build()
            llmSession = LlmInferenceSession.createFromOptions(llmInference, sessionOptions)
        } catch (e: Exception) {
            Log.e("Gemma", "❌ Failed to recreate LLM session", e)
            return
        }

        conversationMemory.clear()
        isGenerating = false
    }


    /** Helper to format stored Q&A pairs into the prompt */
    private fun buildMemoryContext(): String =
        conversationMemory.joinToString("\n") { (q, a) -> "Q: $q\nA: $a" }

    /** Add a Q&A pair to memory, dropping oldest if over limit */
    fun storeMemory(prompt: String, response: String) {
        if (prompt.length < 10 || response.length < 5) return  // skip trivial
        if (conversationMemory.size >= memoryLimit) {
            conversationMemory.removeFirst()
        }
        conversationMemory.add(prompt to response)
    }

    // ---------------------------------
    // TEXT‐ONLY ASYNC GENERATION
    // ---------------------------------
    /**
     * Sends `prompt` (with memory + system prefix) to LLM and streams tokens.
     * onToken(token, done) is invoked for each token; when done==true we store memory & reset.
     */
    fun runTextAsync(prompt: String, onToken: (String, Boolean) -> Unit) {
        cancelOngoingGeneration()

        generationJob = CoroutineScope(Dispatchers.IO).launch {
            isGenerating = true
            val responseBuilder = StringBuilder()
            try {
                // Build the full prompt
                val memoryContext = buildMemoryContext()
                val fullPrompt = """
                    $SYSTEM_PREFIX
                     ${localePrefix()}
                    $memoryContext

                    Q: $prompt
                    A:
                """.trimIndent()

                // Stream generation
                llmSession.addQueryChunk(fullPrompt)
                llmSession.generateResponseAsync { token, done ->
                    responseBuilder.append(token)
                    runCatching { onToken(token, done) }
                    if (done) {
                        // Store into RAG and reset
                        val fullAnswer = responseBuilder.toString().trim()
                        storeMemory(prompt, fullAnswer)
                        conversationMemory.clear()
                        resetSession()
                        isGenerating = false
                    }
                }
            } catch (e: Exception) {
                // Handle errors / token overflow
                val msg = e.message.orEmpty()
                if ("OUT_OF_RANGE" in msg) {
                    Log.e("Gemma", "❌ Context overflow—resetting session", e)
                    resetSession()
                    onToken("[⚠️ Context too long: memory cleared—please retry.]", true)
                } else {
                    Log.e("Gemma", "❌ Text generation failed", e)
                    resetSession()
                    onToken("[⚠️ Error: ${e.message}]", true)
                }
                isGenerating = false
            }
        }
    }

    // ---------------------------------
    // TEXT‐ONLY SYNC GENERATION
    // ---------------------------------
    /**
     * Blocking call to generate a single completion.
     * Respects isBusy to prevent re-entrant calls.
     */
    fun runText(prompt: String): String {
        if (isBusy) return "[⚠️ Busy]"
        isBusy = true
        isGenerating = true
        generationJob?.cancel()

        return try {
            val memoryContext = buildMemoryContext()
            val fullPrompt = "$SYSTEM_PREFIX\n$memoryContext\nQ: $prompt"
            llmSession.addQueryChunk(fullPrompt)
            llmSession.generateResponse() ?: "No response"
        } catch (e: Exception) {
            Log.e("Gemma", "❌ runText failed", e)
            resetSession()
            "⚠️ LLM session error. Try again."
        } finally {
            isBusy = false
        }
    }

    // ---------------------------------
    // IMAGE + PROMPT ASYNC GENERATION
    // ---------------------------------
    /**
     * Sends both an image and text prompt to the LLM, streaming tokens via onToken.
     * Applies the same memory‐augmented prompt logic, auto‐clearing if too long.
     */
    fun runImageWithPromptAsync(
        prompt: String,
        bytes: ByteArray,
        onToken: (String, Boolean) -> Unit
    ) {
        if (isGenerating) {
            Log.w("Gemma", "⚠️ LLM already generating — skipping vision request.")
            onToken("[⚠️ Busy — wait for current task to finish.]", true)
            return
        }

        cancelOngoingGeneration()


        generationJob = CoroutineScope(Dispatchers.IO).launch {
            val responseBuilder = StringBuilder()
            try {
                // Decode the JPEG bytes into MPImage
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                val mpImage = BitmapImageBuilder(bitmap).build()

                // Build memory context
                val memoryContext = buildMemoryContext()

                // Assemble prompt with or without memory
                val basePrompt = if (memoryContext.isNotBlank()) {
                    """
                    $SYSTEM_PREFIX
                    ${localePrefix()}
                    $memoryContext

                    Q: $prompt
                    A:
                    """.trimIndent()
                } else {
                    """
                    $SYSTEM_PREFIX

                    Q: $prompt
                    A:
                    """.trimIndent()
                }

                // Auto-clear memory if prompt too long
                val fullPrompt = if (basePrompt.length > 4_000) {
                    Log.w("Gemma", "🔔 Prompt too big (${basePrompt.length} chars), clearing memory")
                    """
                    $SYSTEM_PREFIX

                    Q: $prompt
                    A:
                    """.trimIndent()
                } else basePrompt

                // Stream generation with image
                llmSession.addQueryChunk(fullPrompt)
                llmSession.addImage(mpImage)
                isGenerating = true
                llmSession.generateResponseAsync { token, done ->
                    runCatching { onToken(token, done) }
                        .onFailure { Log.e("Gemma", "❌ Error during token callback", it) }
                    if (done) {
                        val finalAnswer = responseBuilder.toString().trim()
                        storeMemory(prompt, finalAnswer)
                        conversationMemory.clear()
                        resetSession()
                        isGenerating = false
                    }
                }

                bitmap.recycle()
            } catch (e: Exception) {
                // Handle errors / overflow
                val msg = e.message.orEmpty()
                if ("OUT_OF_RANGE" in msg) {
                    Log.e("Gemma", "❌ Context overflow—resetting session", e)
                    resetSession()
                    onToken("[⚠️ Context too long: memory cleared—please retry.]", true)
                } else {
                    Log.e("Gemma", "❌ Vision generation failed", e)
                    resetSession()
                    onToken("[❌ Error: resetting session. Try again.]", true)
                }
                isGenerating = false
            }
        }
    }

    /** Cancel any currently running generation job and clear flag */
    fun cancelOngoingGeneration() {
        generationJob?.cancel()
        generationJob = null
        isGenerating = false
    }

    /** Cleanly close all underlying resources (LLM & ASR) */
    fun close() {
        generationJob?.cancel()
        try {
            llmInference.close()
        } catch (e: Exception) {
            Log.e("Gemma", "❌ Error closing LLM inference", e)
        }
        asr.close()
    }
}
