package com.example.sightspeak_offline

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.*
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import android.widget.ToggleButton
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.lifecycle.lifecycleScope
import com.example.sightspeak_offline.databinding.ActivityMainBinding
import com.example.sightspeak_offline.hybrid.HybridGemmaHelper
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.*
import java.util.concurrent.Executor


class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "MainActivity"
        private const val MODEL_NAME = "gemma-3n-E2B-it-int4.task"
        private const val TIMEOUT_MS = 10_000L
    }
    // View binding
    private lateinit var binding: ActivityMainBinding

    // The offline LLM + vision helper
    private lateinit var helper: HybridGemmaHelper

    // Text-to-speech engine
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // Flags for ongoing LLM work
    private var isProcessingLLM = false
    private var isFirstChunk = false

    // Path to the local .task model file
    private lateinit var modelFilePath: String


    // Simple RAG memory buffer (last N Q&A pairs)
    private val memoryBuffer = mutableListOf<Pair<String, String>>()
    private val MEMORY_LIMIT = 5

    // CameraX preview and executor
    private lateinit var previewView: PreviewView
    private var imageCapture: ImageCapture? = null
    private var lensFacing = CameraSelector.DEFAULT_BACK_CAMERA
    private lateinit var cameraExecutor: Executor

    // Speech recognizer
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechIntent: Intent

    // UI state: language and mode
    private var selectedLang: String = "English"
    private var isTranslateMode: Boolean = false



    // Launcher for picking a model file if missing
    private val pickModelLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            Log.d(TAG, "Model file picked: $uri")
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            lifecycleScope.launch(Dispatchers.IO) {
                // Copy selected file into app's files/models/
                val dst = File(filesDir, "models").apply { mkdirs() }.resolve(MODEL_NAME)
                contentResolver.openInputStream(it)?.use { inp ->
                    dst.outputStream().use { out -> inp.copyTo(out) }
                }
                initHelper(dst)
            }
        }
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate")
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Wire up the language spinner and mode toggle
        setupLanguageSpinner()
        setupModeToggle()

        // Init TTS
        tts = TextToSpeech(this, this)

        // Camera preview
        previewView = binding.previewView
        cameraExecutor = ContextCompat.getMainExecutor(this)

        //Initialize Speech recognizer
        initSpeechRecognizer()

        // UI buttons
        binding.ttsButton.setOnClickListener {
            val text = binding.inputEditText.text.toString()
            if (isTranslateMode) translateText(text)
            else handleTextPromptAsync(text)

        }
        binding.voiceCmdButton.setOnClickListener { recordAudio() }
        binding.visionButton.setOnClickListener {
            Log.d(TAG, "Vision button clicked")
            if (isTranslateMode) translatePhoto()
            else takePhotoWithPrompt("You are an assistive AI describing scenes to a blind user.\n" +
                    "Use clear, concise, and practical language.\n" +
                    "Name objects if possible (e.g. chair, laptop, window).\n" +
                    "Say where things are, using left, right, top, bottom, or center.\n" +
                    "Avoid vague phrases like \"object\" or \"thing\".\n" +
                    "Avoid poetic, abstract, or emotional language.\n" +
                    "Do not ask questions. Just describe the scene.")
        }
        binding.switchCamButton.setOnClickListener {
            Log.d(TAG, "Switch camera button clicked")
            lensFacing = if (lensFacing == CameraSelector.DEFAULT_BACK_CAMERA)
                CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            startCamera()
        }
        binding.voiceCmdButton.setOnClickListener { recordAudio() }

        // Disable UI until ready
        setUiEnabled(false)

        // Request Permissions & start camera
        requestPermissionsAndStartCamera()

        // Load model
        loadOrPromptModel()
    }

    // ---- TTS initialization callback ----
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            ttsReady = true
            Log.d("TTS", "TTS is ready.")
            tts?.language = Locale.getDefault()
            tts?.setSpeechRate(1.2f)

            val allVoices = tts?.voices
            Log.d("TTS", "🗣️ Available Voices:")
            allVoices?.forEach { voice ->
                Log.d("TTS", "• ${voice.name} | Locale: ${voice.locale} | Gender?: ${voice.name.contains("male", true)}")
            }


            // Try to select a male voice
            val voices = tts?.voices
            val maleVoice = voices?.firstOrNull { voice ->
                voice.locale == Locale.getDefault() && voice.name.contains("male", ignoreCase = true)
            }

            if (maleVoice != null) {
                tts?.voice = maleVoice
                Log.d(TAG, "✅ Male voice selected: ${maleVoice.name}")
            } else {
                Log.w(TAG, "⚠️ No male voice found, using default")
            }


            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    Log.d(TAG, "TTS start $utteranceId")
                }
                override fun onDone(utteranceId: String?) {
                    Log.d(TAG, "TTS done $utteranceId")
                }
                override fun onError(utteranceId: String?) {
                    Log.e(TAG, "TTS error $utteranceId")
                }
            })
            // Greet after a short delay
            lifecycleScope.launch { delay(500); speakOut("Hi I am Murdock, your AI vision assistant, I am getting ready to help you.") }
        } else {
            Log.e(TAG, "TTS init failed: status=$status")
        }
    }

    /** Speak the given text via TTS, with retry on DeadObjectException */
    private fun speakOut(text: String) {
        if (!ttsReady) { Log.w(TAG, "TTS not ready"); return }
        try {
            val uttId = "TTS_${System.currentTimeMillis()}"
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, uttId)
            Log.d(TAG, "Speaking: $text")
        } catch (e: DeadObjectException) {
            Log.e(TAG, "DeadObjectException in TTS, reinit", e)
            reinitializeTts(text)
        } catch (e: Exception) {
            Log.e(TAG, "General TTS error", e)
            reinitializeTts(text)
        }
    }
    /** Reinitialize TTS and retry speaking */
    private fun reinitializeTts(retryText: String? = null) {
        tts?.shutdown()
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ttsReady = true
                tts?.language = Locale.getDefault()
                tts?.setSpeechRate(1.2f)
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        Log.d(TAG, "TTS reinit start $utteranceId")
                    }
                    override fun onDone(utteranceId: String?) {
                        Log.d(TAG, "TTS reinit done $utteranceId")
                    }
                    override fun onError(utteranceId: String?) {
                        Log.e(TAG, "TTS reinit error $utteranceId")
                    }
                })
                Log.d("TTS", "Reinitialized TTS")
                retryText?.let { speakOut(it) }
            } else {
                Log.e("TTS", "Reinit failed: status=$status")
            }
        }
    }

    // ---- UI SETUP HELPERS ----

    /** Populate and handle language selection spinner */

    private var currentLang: String = "English"  // track currently used language

    private fun setupLanguageSpinner() {
        val languages = listOf("English", "中文", "日本語", "한국어", "Deutsch", "Español", "Français")
        val spinner: Spinner = findViewById(R.id.langSpinner)

        // Set custom adapter with white text for dark background
        val adapter = ArrayAdapter(
            this,
            R.layout.spinner_item_white,  // custom selected item layout
            languages
        )
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_white)  // dropdown list style
        spinner.adapter = adapter

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {
                val newLang = languages[pos]
                if (newLang == currentLang) return  // skip if same language

                selectedLang = newLang
                currentLang = newLang
                Log.d(TAG, "Language changed to: $selectedLang")

                // Run model reset on background thread
                if (::helper.isInitialized && ::modelFilePath.isInitialized) {
                    CoroutineScope(Dispatchers.IO).launch {
                        val newHelper = HybridGemmaHelper(this@MainActivity, modelFilePath, selectedLang)
                        withContext(Dispatchers.Main) {
                            helper.close()
                            helper = newHelper
                            Toast.makeText(this@MainActivity, "Language updated", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>) {
                // No-op
            }
        }
    }


    /** Wire up the translate <> Q&A toggle button */
    private fun setupModeToggle() {
        val toggle: ToggleButton = findViewById(R.id.translateToggle)
        toggle.setOnCheckedChangeListener { _, isChecked ->
            isTranslateMode = isChecked
            val msg = if (isChecked) "Translate mode on" else "Question and answer mode on"
            Log.d(TAG, msg)
            if (ttsReady) speakOut(msg)
        }
    }

    /** Request camera & audio permissions, then start camera */
    private fun requestPermissionsAndStartCamera() {
        val perms = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        val toReq = perms.filter {
            ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (toReq.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, toReq.toTypedArray(), 0)
        } else {
            startCamera()
        }
    }


    /** Prepare or prompt for model file */
    private fun loadOrPromptModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            val modelFile = File(filesDir, "models").apply { mkdirs() }.resolve(MODEL_NAME)
            if (!modelFile.exists()) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Model missing: pick one.", Toast.LENGTH_LONG).show()
                    pickModelLauncher.launch(arrayOf("*/*"))
                }
            } else {
                initHelper(modelFile)
            }
        }
    }

    /** Initialize the HybridGemmaHelper on IO, then enable UI on Main */
    private suspend fun initHelper(modelFile: File) {
        Log.d(TAG, "Initializing helper with ${modelFile.name}")
        modelFilePath = modelFile.absolutePath
        helper = HybridGemmaHelper(this, modelFilePath, selectedLang)
        withContext(Dispatchers.Main) {
            setUiEnabled(true)
            Toast.makeText(this@MainActivity, "Ready", Toast.LENGTH_SHORT).show()
        }
    }

    /** Enable/disable all main UI buttons */
    private fun setUiEnabled(enabled: Boolean) {
        listOf(
            binding.ttsButton,
            binding.visionButton,
            binding.switchCamButton,
            binding.voiceCmdButton
        ).forEach { it.isEnabled = enabled }
    }

    // ---------------------
    // TIMEOUT HELPER
    // ---------------------
    /**
     * Runs a 10s watchdog. If an LLM call is still in progress
     * after TIMEOUT_MS, it cancels and calls onTimeout().
     */
    private fun withGenerationTimeout(onTimeout: () -> Unit) {
        lifecycleScope.launch {
            delay(TIMEOUT_MS)
            if (isProcessingLLM) {
                isProcessingLLM = false
                Log.w(TAG, "⚠️ Generation timeout")
                safeResetSession()
                onTimeout()
            }
        }
    }

    /** Safely cancel & reset the helper session */
    private fun safeResetSession() {
        if (isProcessingLLM) {
            helper.cancelOngoingGeneration()
            isProcessingLLM = false
        }
        helper.resetSession()
    }

    // ---------------------
    // VISION VS TEXT DECIDER
    // ---------------------
    /**
     * Sends a small prompt to the LLM to decide if visual input is needed.
     * Calls either onVision() or onText().
     */
    private fun decideVision(
        prompt: String,
        onVision: () -> Unit,
        onText:   () -> Unit
    ) {
        Log.d(TAG, "Deciding vision for prompt: $prompt")
        val decisionPrompt = """
      You are SightSpeak, an offline, privacy-first, multimodal mobile assistant for blind users.
      Decide if the following prompt needs visual input. Respond “yes” or “no”.

      Q: What is on the table?
      A: yes

      Q: Who is speaking in this audio?
      A: yes

      Q: What does this mean in text?
      A: no
      
      Q: Where are shoes?
      A: yes

      Q: $prompt
      A:
    """.trimIndent()

        val buffer = StringBuilder()
        helper.runTextAsync(decisionPrompt) { token, done ->
            runOnUiThread {
                if (!done) {
                    buffer.append(token); return@runOnUiThread
                }
                val raw = buffer.toString()
                    .replace(Regex("(?i)^(A:|Answer:)"), "")
                    .trim().lowercase(Locale.ROOT)
                val first = raw.split("\\s+".toRegex()).firstOrNull() ?: "no"
                val decision = if (first == "yes") "yes" else "no"
                Log.d(TAG, "Decision=$decision from raw='$raw'")
                safeResetSession()
                when (decision) {
                    "yes" -> onVision()
                    else  -> onText()
                }
            }
        }

        // Start timeout watchdog
        withGenerationTimeout {
            runOnUiThread {
                binding.ttsButton.isEnabled = true
                Toast.makeText(this, "Be patient, still processing", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ---------------------
    // TEXT-ONLY PATH
    // ---------------------
    /**
     * Entry for text input from the UI button.
     * Detects greetings, then uses decideVision().
     */
    private fun handleTextPromptAsync(prompt: String) {
        if (isProcessingLLM) {
            helper.cancelOngoingGeneration()
            Log.w(TAG, "Cancel previous text generation")
        }
        safeResetSession()
        isProcessingLLM = true
        isFirstChunk = false
        binding.resultText.text = ""

        val safePrompt = prompt.take(200)
        val lower = safePrompt.lowercase(Locale.ROOT)
        // If greeting, skip vision decision
        if (lower.matches(Regex("^(hi|hello|hey|how are you)(\\b.*)?"))) {
            Log.d(TAG, "Greeting detected, skip vision")
            streamingTextWithSpeech(safePrompt)
            return
        }

        binding.resultText.text = ""
        decideVision(
            prompt = safePrompt,
            onVision = { takePhotoWithPrompt(safePrompt) },
            onText   = { streamingTextWithSpeech(safePrompt) }
        )
    }

    /**
     * Streams LLM response tokens, updates UI, and speaks in chunks.
     * Adds final answer to RAG memory.
     */
    private fun streamingTextWithSpeech(prompt: String) {
        isProcessingLLM = true
        isFirstChunk = true
        binding.resultText.text = ""              // clear the UI
        val allText = StringBuilder()
        var buffer = StringBuilder()

        // pattern to strip any leading "A:" or "Answer:"
        val stripPromptLabel = Regex("(?i)^(?:A:|Answer:)\\s*")

        // pattern to grab up through the first sentence-ending punctuation.
        val sentencePattern = Regex("""(?<=[.!?])\s+""")
        val fullPrompt = buildRagPrompt(prompt)
        Log.d(TAG, "▶️ streamingTextWithSpeech: sending LLM prompt")

        helper.runTextAsync(fullPrompt) { token, done ->
            runOnUiThread {
                // **always** strip any A: or Answer: prefix
                val clean = token.replaceFirst(stripPromptLabel, "")

                allText.append(clean)
                binding.resultText.append(clean)
                buffer.append(clean)

                val bufStr = buffer.toString()

                // 1) Speak full sentences immediately
                val sentenceMatch = Regex("^(.*?[.!?])(?:\\s|$)").find(bufStr)

                sentenceMatch?.let {
                    val sentence = it.groupValues[1]
                    speakOut(sentence.trim())
                    Log.d("TTS", "🔊 Spoke sentence: $sentence")
                    buffer = StringBuilder(bufStr.substring(it.range.last + 1))
                }


                }

                if (done) {
                    isProcessingLLM = false
                    // finally speak any remaining fragment
                    val tail = buffer.toString().trim()
                    if (tail.isNotBlank()) {
                        speakOut(tail)
                        Log.d("TTS", "🔊 Spoke tail chunk: $tail")
                    }
                    // save to memory
                    val finalAnswer = allText.toString().trim().replace(Regex("(?i)^A:\\s*"), "")   // Strip leading A:
                    memoryBuffer.add(prompt to finalAnswer)
                    if (memoryBuffer.size > MEMORY_LIMIT) memoryBuffer.removeAt(0)
                    binding.ttsButton.isEnabled = true
                }
            }


        // Timeout safeguard
        withGenerationTimeout {
            runOnUiThread {
                binding.ttsButton.isEnabled = true
                Toast.makeText(this, "Be patient: Still processing.", Toast.LENGTH_SHORT).show()
            }
        }
    }


    /**
     * Prepends RAG memory if under length limit; otherwise clears it.
     * Also adds language prefix for consistency.
     */
    /**
     * Prepends RAG memory if under length limit; otherwise clears it.
     * Also adds language prefix for consistency.
     */
    private fun buildRagPrompt(userPrompt: String): String {
        // Language-specific prefix
        val prefix = when (selectedLang) {
            "中文"     -> "Please answer the following question **in Simplified Chinese**:"
            "日本語"   -> "Please answer the following question **in Japanese**:"
            "한국어"   -> "Please answer the following question **in Korean**:"
            "Deutsch" -> "Please answer the following question **in German**:"
            "Español" -> "Please answer the following question **in Spanish**"
            "Français"-> "Please answer the following question **in French**:"
            else      -> "Please answer the following in English:"
        }

        // Join RAG memory context
        val memoryText = memoryBuffer.joinToString("\n") { (q, a) -> "Q: $q\nA: $a" }

        // Build the main prompt body
        val promptBody = if (memoryText.isNotBlank()) {
            "You are SightSpeak, an offline assistant. Use the following memory to help answer:\n$memoryText\n\nQ: $userPrompt\nA:"
        } else {
            "Q: $userPrompt\nA:"
        }

        // Truncate memory if too large
        val fullPrompt = "$prefix\n$promptBody"
        return if (fullPrompt.length > 4000) {
            Log.w(TAG, "Context too big, clearing memory")
            memoryBuffer.clear()
            "$prefix\nQ: $userPrompt\nA:"
        } else {
            Log.d(TAG, "📨 Prompt to LLM:\n$fullPrompt")
            fullPrompt
        }
    }

    // ---------------------
    // TRANSLATION PATH
    // ---------------------
    /**
     * Streams a translation LLM prompt and speaks the result.
     */
    private fun translateText(input: String) {
        if (isProcessingLLM) helper.cancelOngoingGeneration()
        safeResetSession()
        isProcessingLLM = true
        isFirstChunk = true
        binding.resultText.text = ""

        // Build translate instruction
        val instr = when (selectedLang) {
            "中文"    -> "Translate this following text to Chinese.Output only the translated text; do not add any commentary or follow-up questions:"
            "日本語"  -> "Translate the following text to Japanese.Output only the translated text; do not add any commentary or follow-up questions:"
            "한국어"  -> "Translate the following text to Korean.Output only the translated text; do not add any commentary or follow-up questions:"
            "Deutsch" -> "Translate the following text to German.Output only the translated text; do not add any commentary or follow-up questions:"
            "Español" -> "Translate the following text to Spanish.Output only the translated text; do not add any commentary or follow-up questions:"
            "Français"-> "Translate the following text to French.Output only the translated text; do not add any commentary or follow-up questions:"
            else      -> "Translate this following text to English.Output only the translated text; do not add any commentary or follow-up questions:"
        }
        val prompt = "$instr\n\"$input\"\nA:"
        val allText = StringBuilder()
        var buffer = StringBuilder()

        helper.runTextAsync(prompt) { token, done ->
            runOnUiThread {
                allText.append(token)
                binding.resultText.append(token)
                buffer.append(token)
                if ((buffer.split("\\s+".toRegex()).size >= 20 || done) && buffer.isNotBlank()) {
                    speakOut(buffer.toString().trim()); buffer = StringBuilder()
                }
                if (done) {
                    isProcessingLLM = false
                    if (buffer.isNotBlank()) speakOut(buffer.toString().trim())
                    binding.ttsButton.isEnabled = true
                }
            }
        }

        withGenerationTimeout {
            runOnUiThread {
                binding.ttsButton.isEnabled = true
                Toast.makeText(this, "Translate took too long, please retry.", Toast.LENGTH_SHORT).show()
            }
        }
    }


//OCR
    private fun runOcrOnBitmap(bitmap: Bitmap, onResult: (String) -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val recognizer = when (selectedLang) {
        "中文" -> TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())
        "日本語" -> TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())
        "한국어" -> TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        else -> TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) // For English, French, etc.
    }

    recognizer.process(image)
            .addOnSuccessListener { result ->
                val text = result.text
                Log.d(TAG, "OCR result: $text")
                onResult(text)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR failed", e)
                Toast.makeText(this, "OCR failed", Toast.LENGTH_SHORT).show()
            }
    }

    /**
     * Captures the current preview frame, buffers the description
     * fully, then calls translateText().
     */
    private fun translatePhoto() {
        val bitmap = binding.previewView.bitmap ?: return
        val scaled = bitmap.scale(240, 160)
        runOcrOnBitmap(scaled) { extractedText ->
            if (extractedText.isBlank()) {
                Toast.makeText(this, "No text detected in the image.", Toast.LENGTH_SHORT).show()
                return@runOcrOnBitmap
            }
            translateText(extractedText)
        }

    withGenerationTimeout {
            runOnUiThread {
                binding.ttsButton.isEnabled = true
                Toast.makeText(this, "Be patient: Still processing.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ---------------------
    // IMAGE→PROMPT PATH
    // ---------------------
    /**
     * Capture a frame, send it + prompt to the LLM, and speak the result.
     */
    private fun takePhotoWithPrompt(prompt: String) {
        val bitmap = binding.previewView.bitmap ?: return
        Log.d(TAG, "Capturing photo for prompt: $prompt")
        if (isProcessingLLM) {
            helper.cancelOngoingGeneration()
            Log.w(TAG, "Cancel previous image generation")
        }
        safeResetSession()
        isProcessingLLM = true
        isFirstChunk = false
        binding.resultText.text = ""


        // Compress and send

        val baos = ByteArrayOutputStream()
        val scaled = bitmap.scale(92,72)
        scaled.compress(Bitmap.CompressFormat.JPEG,40,baos)
        val bytes = baos.toByteArray()

        bitmap.recycle(); scaled.recycle()

        // Clean and expand prompt
        val basePrompt = prompt.trim().ifBlank {
            ("You are an assistive AI describing scenes to a blind user.\n" +
                    "Use clear, concise, and practical language.\n" +
                    "Name objects if possible (e.g. chair, laptop, window).\n" +
                    "Say where things are, using left, right, top, bottom, or center.\n" +
                    "Avoid vague phrases like \"object\" or \"thing\".\n" +
                    "Avoid poetic, abstract, or emotional language.\n" +
                    "Do not ask questions. Just describe the scene.")
        }

        // Build RAG prompt and send
        val fullPrompt = buildRagPrompt(basePrompt)
        helper.runImageWithPromptAsync(fullPrompt, bytes) { token, done ->
            runOnUiThread {
                if (!isFirstChunk) {
                    binding.resultText.text = ""
                    isFirstChunk = true
                }
                binding.resultText.append(token)
            }
            if (done) {
                val answer = binding.resultText.text.toString().trim(). replace(Regex("(?i)^A:\\s*"), "")
                Log.d(TAG, "Image analysis result: $answer")
                runOnUiThread {
                    speakOut(answer)
                    binding.ttsButton.isEnabled = true
                }
                isProcessingLLM = false
            }
        }

        withGenerationTimeout {
            runOnUiThread {
                binding.ttsButton.isEnabled = true
                Toast.makeText(this, "Be patient, still processing.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ---------------------
    // AUDIO→TEXT PATH
    // ---------------------
    /** Start speech recognition with the appropriate locale */
    private fun recordAudio() {
        Log.d(TAG, "Starting speech recognition")
        speechRecognizer.cancel()
        val localeTag = when (selectedLang) {
            "中文" -> "zh-CN"
            "日本語"-> "ja-JP"
            "한국어"-> "ko-KR"
            "Deutsch"-> "de-DE"
            "Español"-> "es-ES"
            "Français"-> "fr-FR"
            else     -> "en-US"
        }
        speechIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, localeTag)
        speechRecognizer.startListening(speechIntent)
        binding.voiceCmdButton.isEnabled = false
    }

    /** Initialize the Android SpeechRecognizer and its listener */
    private fun initSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) { Log.d(TAG, "onReadyForSpeech") }
                override fun onBeginningOfSpeech() { Log.d(TAG, "onBeginningOfSpeech") }
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() { Log.d(TAG, "onEndOfSpeech") }
                override fun onEvent(eventType: Int, params: Bundle?) {}
                override fun onError(error: Int) {
                    Log.e(TAG, "Speech error: $error")
                    runOnUiThread {
                        if (error == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED) {
                            Toast.makeText(this@MainActivity, "$selectedLang not supported; falling back to English", Toast.LENGTH_LONG).show()
                            speechIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                            startListening(speechIntent)
                        } else {
                            Toast.makeText(this@MainActivity, "Speech error $error", Toast.LENGTH_SHORT).show()
                            binding.voiceCmdButton.isEnabled = true
                        }
                    }
                }
                override fun onPartialResults(partial: Bundle?) {}
                override fun onResults(results: Bundle?) {
                    val cmd = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull().orEmpty()
                    Log.d(TAG, "Voice command: $cmd")
                    handleVoiceCommand(cmd)
                    runOnUiThread { binding.voiceCmdButton.isEnabled = true }
                }
            })
        }

        speechIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }
    }

    /** Handle the recognized voice command, similar to text path */
    private fun handleVoiceCommand(cmd: String) {
        if (isTranslateMode) {
            translateText(cmd)
            return
        }
        if (isProcessingLLM) {
            helper.cancelOngoingGeneration()
            Log.w(TAG, "Cancel previous voice generation")
        }
        safeResetSession()
        isProcessingLLM = true
        isFirstChunk = false


        val safePrompt = cmd.take(200)
        if (safePrompt.lowercase(Locale.ROOT).matches(Regex("^(hi|hello|hey)(\\b.*)?"))) {
            Log.d(TAG, "Greeting detected, skip vision")
            streamingTextWithSpeech(safePrompt)
            return
        }

        binding.resultText.text = ""
        decideVision(
            prompt = safePrompt,
            onVision = { takePhotoWithPrompt(safePrompt) },
            onText   = { streamingTextWithSpeech(safePrompt) }
        )
    }

    // ---------------------
    // CAMERA PREVIEW SETUP
    // ---------------------
    private fun startCamera() {
        Log.d(TAG, "Starting camera")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(320, 240))
                .build()

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, lensFacing, preview, imageCapture)
        }, cameraExecutor)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy")
        speechRecognizer.destroy()
        tts?.shutdown()
        if (::helper.isInitialized) helper.close()
    }
}