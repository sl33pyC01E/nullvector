plugins { id("com.android.application") }

android {
    namespace = "world.nullvector.mobile"
    compileSdk = 36
    defaultConfig {
        applicationId = "world.nullvector.mobile"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    buildTypes { release { isMinifyEnabled = false } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += "onnx" }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
}
