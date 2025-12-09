plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.expensetracker"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.expensetracker"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        viewBinding = true
    }
}

// Exclude litert-api from all configurations to avoid conflicts with tensorflow-lite
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.lifecycle.livedata.ktx)
    implementation(libs.lifecycle.viewmodel.ktx)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    
    // TensorFlow Lite for SMS analysis
    // Version 2.17.0 is required for FULLY_CONNECTED opcode version 12 support
    // However, 2.17.0 has a known issue with InterpreterApi class availability
    // Workaround: Use 2.17.0 runtime but compile against 2.15.0 API, then use reflection
    // OR: Wait for TensorFlow Lite 2.17.1+ fix, or use nightly/snapshot builds
    implementation("org.tensorflow:tensorflow-lite-api:2.15.0")
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    
    // Retrofit for API calls
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
    
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}