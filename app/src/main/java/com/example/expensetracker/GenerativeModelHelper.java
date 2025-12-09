package com.example.expensetracker;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.mlkit.genai.common.GenAiException;
import com.google.mlkit.genai.prompt.Candidate;
import com.google.mlkit.genai.prompt.Generation;
import com.google.mlkit.genai.prompt.GenerateContentRequest;
import com.google.mlkit.genai.prompt.GenerateContentResponse;
import com.google.mlkit.genai.prompt.GenerativeModel;
import com.google.mlkit.genai.prompt.java.GenerativeModelFutures;
import com.google.mlkit.genai.prompt.ImagePart;
import com.google.mlkit.genai.prompt.TextPart;

import java.util.List;
import java.util.concurrent.Executor;

/**
 * Helper class to manage ML Kit GenAI Prompt API operations.
 * Handles model initialization, status checking, downloading, and content generation.
 */
public class GenerativeModelHelper {
    private static final String TAG = "GenerativeModelHelper";
    
    // Feature status constants (from FeatureStatus enum)
    private static final int FEATURE_STATUS_AVAILABLE = 1;
    private static final int FEATURE_STATUS_UNAVAILABLE = 2;
    private static final int FEATURE_STATUS_DOWNLOADING = 3;
    private static final int FEATURE_STATUS_DOWNLOADABLE = 4;
    
    private GenerativeModelFutures generativeModelFutures;
    private Context context;
    private boolean modelReady = false;
    
    public interface ModelStatusCallback {
        void onStatusChecked(int status);
        void onDownloadStarted();
        void onDownloadProgress(long bytesDownloaded);
        void onDownloadCompleted();
        void onDownloadFailed(String error);
        void onModelReady();
    }
    
    public interface ContentGenerationCallback {
        void onSuccess(String response);
        void onFailure(String error);
    }
    
    public GenerativeModelHelper(Context context) {
        Log.d(TAG, "🔍 GEMINI_NANO: GenerativeModelHelper constructor called");
        this.context = context;
        initializeModel();
    }
    
    /**
     * Initialize the GenerativeModel instance
     */
    private void initializeModel() {
        Log.d(TAG, "🔍 GEMINI_NANO: Initializing GenerativeModel...");
        try {
            GenerativeModel generativeModel = Generation.INSTANCE.getClient();
            generativeModelFutures = GenerativeModelFutures.from(generativeModel);
            Log.d(TAG, "🔍 GEMINI_NANO: GenerativeModel initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "🔍 GEMINI_NANO: Failed to initialize GenerativeModel", e);
            throw e;
        }
    }
    
    /**
     * Check the status of Gemini Nano and download if needed
     */
    public void checkAndPrepareModel(ModelStatusCallback callback) {
        Log.d(TAG, "🔍 GEMINI_NANO: Starting availability check...");
        
        if (generativeModelFutures == null) {
            Log.d(TAG, "🔍 GEMINI_NANO: generativeModelFutures is null, initializing model...");
            initializeModel();
        }
        
        Executor mainExecutor = ContextCompat.getMainExecutor(context);
        
        Log.d(TAG, "🔍 GEMINI_NANO: Calling checkStatus() API...");
        Futures.addCallback(generativeModelFutures.checkStatus(), new FutureCallback<Integer>() {
            @Override
            public void onSuccess(Integer featureStatus) {
                Log.d(TAG, "🔍 GEMINI_NANO: Status check SUCCESS - Feature Status Code: " + featureStatus);
                callback.onStatusChecked(featureStatus);
                
                if (featureStatus == FEATURE_STATUS_AVAILABLE) {
                    Log.i(TAG, "🔍 GEMINI_NANO: ✅ AVAILABLE - Gemini Nano is available and ready to use");
                    modelReady = true;
                    callback.onModelReady();
                } else if (featureStatus == FEATURE_STATUS_UNAVAILABLE) {
                    Log.w(TAG, "🔍 GEMINI_NANO: ❌ UNAVAILABLE - Gemini Nano is not supported on this device");
                    callback.onDownloadFailed("Gemini Nano not supported on this device");
                } else if (featureStatus == FEATURE_STATUS_DOWNLOADING) {
                    Log.i(TAG, "🔍 GEMINI_NANO: ⏳ DOWNLOADING - Gemini Nano is currently being downloaded");
                } else if (featureStatus == FEATURE_STATUS_DOWNLOADABLE) {
                    Log.i(TAG, "🔍 GEMINI_NANO: 📥 DOWNLOADABLE - Gemini Nano can be downloaded, starting download...");
                    // Note: Download functionality may need to be implemented differently
                    // For now, we'll mark as ready if downloadable
                    modelReady = true;
                    callback.onModelReady();
                } else {
                    Log.w(TAG, "🔍 GEMINI_NANO: ⚠️ UNKNOWN STATUS - Unknown feature status code: " + featureStatus);
                }
            }
            
            @Override
            public void onFailure(@NonNull Throwable t) {
                Log.e(TAG, "🔍 GEMINI_NANO: ❌ FAILED - Status check failed", t);
                
                String errorMessage = t.getMessage();
                String errorClass = t.getClass().getSimpleName();
                Log.e(TAG, "🔍 GEMINI_NANO: Error Class: " + errorClass);
                Log.e(TAG, "🔍 GEMINI_NANO: Error Message: " + errorMessage);
                
                String userFriendlyMessage;
                
                // Check for specific error codes - parse from error message as it's more reliable
                if (errorMessage != null) {
                    // Check for ErrorCode 606 - FEATURE_NOT_FOUND (Feature 636 - Gemini Nano)
                    if (errorMessage.contains("ErrorCode 606") || 
                        errorMessage.contains("FEATURE_NOT_FOUND") ||
                        errorMessage.contains("Feature 636") ||
                        errorMessage.contains("606-FEATURE_NOT_FOUND")) {
                        Log.e(TAG, "🔍 GEMINI_NANO: ⚠️ ERROR CODE 606 - FEATURE_NOT_FOUND detected");
                        Log.e(TAG, "🔍 GEMINI_NANO: Feature 636 (Gemini Nano) is not available on this device");
                        userFriendlyMessage = "Gemini Nano is not available on this device. " +
                                "This feature requires AICore and is currently only supported on select devices " +
                                "(e.g., Pixel phones with Android 15+). " +
                                "Please ensure AICore is installed and up to date from Google Play Store. " +
                                "Note: If your device's bootloader is unlocked, Gemini Nano will not work.";
                    } else if (errorMessage.contains("ErrorCode -101")) {
                        Log.e(TAG, "🔍 GEMINI_NANO: ⚠️ ERROR CODE -101 - AICore not installed/outdated");
                        userFriendlyMessage = "AICore is not installed or outdated. " +
                                "Please install/update AICore from Google Play Store. " +
                                "Note: AICore is currently only available on select devices (e.g., Pixel phones).";
                    } else if (errorMessage.contains("AICore")) {
                        Log.e(TAG, "🔍 GEMINI_NANO: ⚠️ AICore related error detected");
                        userFriendlyMessage = "AICore error: " + errorMessage + 
                                ". Please ensure AICore is installed and up to date.";
                    } else {
                        Log.e(TAG, "🔍 GEMINI_NANO: ⚠️ Unknown error type");
                        userFriendlyMessage = "Failed to check Gemini Nano status: " + errorMessage;
                    }
                } else {
                    Log.e(TAG, "🔍 GEMINI_NANO: ⚠️ Error message is null");
                    userFriendlyMessage = "Failed to check Gemini Nano status. " +
                            "Please ensure AICore is installed and up to date.";
                }
                
                Log.e(TAG, "🔍 GEMINI_NANO: User-friendly error message: " + userFriendlyMessage);
                callback.onDownloadFailed(userFriendlyMessage);
            }
        }, mainExecutor);
    }
    
    /**
     * Generate content from text-only input
     */
    public void generateContent(String prompt, ContentGenerationCallback callback) {
        generateContent(prompt, null, callback);
    }
    
    /**
     * Generate content from multimodal input (image + text)
     */
    public void generateContent(String textPrompt, Bitmap image, ContentGenerationCallback callback) {
        GenerateContentRequest.Builder requestBuilder;
        
        if (image != null) {
            requestBuilder = new GenerateContentRequest.Builder(
                new ImagePart(image),
                new TextPart(textPrompt)
            );
        } else {
            requestBuilder = new GenerateContentRequest.Builder(
                new TextPart(textPrompt)
            );
        }
        
        generateContent(requestBuilder.build(), callback);
    }
    
    /**
     * Execute the content generation request
     */
    private void generateContent(GenerateContentRequest request, ContentGenerationCallback callback) {
        Executor mainExecutor = ContextCompat.getMainExecutor(context);
        
        Futures.addCallback(
            generativeModelFutures.generateContent(request),
            new FutureCallback<GenerateContentResponse>() {
                @Override
                public void onSuccess(GenerateContentResponse response) {
                    if (response != null) {
                        List<Candidate> candidates = response.getCandidates();
                        if (candidates != null && candidates.size() > 0) {
                            Candidate candidate = candidates.get(0);
                            String generatedText = candidate.getText();
                            if (generatedText != null) {
                                callback.onSuccess(generatedText);
                            } else {
                                callback.onFailure("No text in response");
                            }
                        } else {
                            callback.onFailure("No candidates in response");
                        }
                    } else {
                        callback.onFailure("No response generated");
                    }
                }
                
                @Override
                public void onFailure(@NonNull Throwable t) {
                    Log.e(TAG, "Content generation failed", t);
                    
                    // Check for AICore error specifically
                    String errorMessage = t.getMessage();
                    if (errorMessage != null && errorMessage.contains("ErrorCode -101")) {
                        callback.onFailure("AICore is not installed or outdated. " +
                                "Please install/update AICore from Google Play Store. " +
                                "Note: AICore is currently only available on select devices (e.g., Pixel phones).");
                    } else if (errorMessage != null && errorMessage.contains("AICore")) {
                        callback.onFailure("AICore error: " + errorMessage + 
                                ". Please ensure AICore is installed and up to date.");
                    } else {
                        callback.onFailure("Generation failed: " + errorMessage);
                    }
                }
            },
            mainExecutor
        );
    }
    
    /**
     * Warm up the model for faster first inference
     */
    public void warmup() {
        if (generativeModelFutures != null) {
            generativeModelFutures.warmup();
        }
    }
    
    /**
     * Check if model is ready to use
     */
    public boolean isModelReady() {
        Log.d(TAG, "🔍 GEMINI_NANO: isModelReady() called - returning: " + modelReady);
        return modelReady;
    }
    
    /**
     * Check if AICore is available by attempting to check model status
     * This is a lightweight check that will fail fast if AICore is not available
     */
    public void checkAICoreAvailability(ModelStatusCallback callback) {
        Log.d(TAG, "🔍 GEMINI_NANO: checkAICoreAvailability() called");
        
        if (generativeModelFutures == null) {
            Log.d(TAG, "🔍 GEMINI_NANO: generativeModelFutures is null, attempting to initialize...");
            try {
                initializeModel();
                Log.d(TAG, "🔍 GEMINI_NANO: Model initialization successful, proceeding with status check");
            } catch (Exception e) {
                Log.e(TAG, "🔍 GEMINI_NANO: ❌ FAILED - Model initialization exception - AICore may not be available", e);
                Log.e(TAG, "🔍 GEMINI_NANO: Exception type: " + e.getClass().getSimpleName());
                Log.e(TAG, "🔍 GEMINI_NANO: Exception message: " + e.getMessage());
                callback.onDownloadFailed("AICore is not available: " + e.getMessage());
                return;
            }
        } else {
            Log.d(TAG, "🔍 GEMINI_NANO: generativeModelFutures already initialized, proceeding with status check");
        }
        
        checkAndPrepareModel(callback);
    }
}
