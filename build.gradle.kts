plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.sightspeak_offline"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.sightspeak_offline"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // NPU / LiteRT: arm64 only
        ndk { abiFilters += listOf("arm64-v8a") }

        // Qualcomm runtimes sometimes need this legacy packaging
        packagingOptions {
            jniLibs.useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            // optional debug settings
        }
    }

    buildFeatures {
        viewBinding = true
    }

    // Keep model files uncompressed if you bundle them
    androidResources {
        noCompress += listOf("task", "tflite", "bin", "gguf", "litertlm")
    }

    // Java/Kotlin
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
            languageVersion.set(org.jetbrains.kotlin.gradle.dsl.KotlinVersion.KOTLIN_2_1)
        }
    }

    // Drop the accidental Compose theme package
    sourceSets["main"].java.srcDirs("src/main/java")
    sourceSets["main"].java.exclude("com/example/sightspeak_offline/ui/theme/**")

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

configurations.all {
    resolutionStrategy {
        // force both artifacts to 2.12.0 so they don’t clash
        force(
            "org.tensorflow:tensorflow-lite:2.12.0",
            "org.tensorflow:tensorflow-lite-api:2.12.0"
        )
    }
}


dependencies {
    // AndroidX
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.7.3")

    // CameraX
    val camerax = "1.4.2"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ---- MediaPipe Tasks ----

    implementation("com.google.mediapipe:tasks-vision:0.10.26")
    implementation("com.google.mediapipe:tasks-text:0.10.26")
    implementation("com.google.mediapipe:tasks-audio:0.10.26")
    implementation("com.google.mediapipe:tasks-genai:0.10.25")
    implementation("com.google.mediapipe:tasks-core:0.10.26")
    implementation("com.google.ai.edge.litert:litert:2.0.1-alpha")
    implementation("com.google.android.gms:play-services-tflite-acceleration-service:16.0.0-beta01")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.0")
    implementation("com.google.mlkit:text-recognition:16.0.0-beta5")
    implementation("com.google.mlkit:text-recognition-chinese:16.0.0-beta5")
    implementation("com.google.mlkit:text-recognition-japanese:16.0.0-beta5")
    implementation("com.google.mlkit:text-recognition-korean:16.0.0-beta5")




    // LiteRT core
    implementation("com.google.ai.edge.litert:litert:2.0.1-alpha")



    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
