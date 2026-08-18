plugins { id("com.android.application") }

val emulatorAbi = providers.gradleProperty("emulatorAbi").orNull == "true"

android {
    namespace = "world.nullvector.mobile"
    compileSdk = 36
    defaultConfig {
        applicationId = "world.nullvector.mobile"
        minSdk = 29
        targetSdk = 36
        versionCode = 12
        versionName = "0.8.0-neural-gpu"
        ndk { abiFilters += if (emulatorAbi) "x86_64" else "arm64-v8a" }
    }
    flavorDimensions += "runtime"
    productFlavors {
        create("fp32") {
            dimension = "runtime"
            applicationIdSuffix = ".fp32"
            versionNameSuffix = "-fp32"
            buildConfigField("boolean", "SPLIT_ACTION", "false")
        }
        create("int8") {
            dimension = "runtime"
            applicationIdSuffix = ".int8"
            versionNameSuffix = "-int8"
            buildConfigField("boolean", "SPLIT_ACTION", "true")
        }
    }
    buildFeatures { buildConfig = true }
    buildTypes {
        release {
            isMinifyEnabled = false
            // Preview releases are development-signed so they can be sideloaded.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += listOf("onnx", "tflite") }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
    implementation("com.google.android.gms:play-services-tflite-java:16.5.0")
    implementation("com.google.android.gms:play-services-tflite-gpu:16.5.0")
}
