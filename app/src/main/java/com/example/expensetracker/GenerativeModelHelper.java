package com.example.expensetracker;

import android.content.Context;
import android.util.Log;

import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX helper: loads sms_model.onnx and runs inference on SMS text.
 * 
 * Requirement: assets/sms_model.onnx
 */
public class GenerativeModelHelper {
    private static final String TAG = "GenerativeModelHelper";
    private static final float CONFIDENCE_THRESHOLD = 0.5f;

    private final Context context;
    private OrtEnvironment env;
    private OrtSession session;
    private volatile boolean modelReady = false;
    private volatile boolean isLoading = false;
    private ModelStatusCallback pendingCallback = null;
    private int inputVectorSize = -1; // Detected from model

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
        this.context = context;
        initializeModels();
    }

    private void initializeModels() {
        isLoading = true;
        // Use background thread to avoid blocking UI thread (prevents ANR)
        Executor backgroundExecutor = Executors.newSingleThreadExecutor();
        Executor mainExecutor = ContextCompat.getMainExecutor(context);
        
        backgroundExecutor.execute(() -> {
            try {
                env = OrtEnvironment.getEnvironment();

                byte[] modelBytes = loadAssetBytes("sms_model.onnx");
                session = env.createSession(modelBytes);

                // Detect input tensor shape from model
                detectInputShape();

                modelReady = true;
                isLoading = false;
                Log.d(TAG, "✅ ONNX model loaded, input size: " + inputVectorSize);
                
                // Notify pending callback on main thread if any
                if (pendingCallback != null) {
                    ModelStatusCallback callback = pendingCallback;
                    pendingCallback = null;
                    mainExecutor.execute(() -> {
                        callback.onStatusChecked(1); // available
                        callback.onModelReady();
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Failed to load ONNX model", e);
                modelReady = false;
                isLoading = false;
                
                // Notify pending callback of failure on main thread
                if (pendingCallback != null) {
                    ModelStatusCallback callback = pendingCallback;
                    pendingCallback = null;
                    mainExecutor.execute(() -> {
                        callback.onStatusChecked(2); // unavailable
                        callback.onDownloadFailed("Failed to load ONNX model: " + e.getMessage());
                    });
                }
            }
        });
    }

    /**
     * Detect input tensor shape from the model.
     * Since ONNX Runtime Android API doesn't provide direct shape access,
     * we use a default size and allow dynamic adjustment.
     */
    private void detectInputShape() throws OrtException {
        if (session == null) {
            throw new IllegalStateException("Session not initialized");
        }

        Set<String> inputNames = session.getInputNames();
        if (inputNames.isEmpty()) {
            throw new IllegalStateException("Model has no inputs");
        }

        // ONNX Runtime Android doesn't provide direct access to input metadata
        // Use a reasonable default size (128) for SMS text encoding
        // This can be adjusted based on your model's expected input size
        inputVectorSize = 128; // Default size
        
        Log.d(TAG, "Using default input vector size: " + inputVectorSize + " (model has " + inputNames.size() + " input(s))");
        
        // Optional: Try to infer size from a test run if needed
        // For now, we'll use the default and let padding handle shorter texts
    }

    private byte[] loadAssetBytes(String name) throws IOException {
        InputStream is = context.getAssets().open(name);
        try {
            int size = is.available();
            byte[] b = new byte[size];
            int totalRead = 0;
            while (totalRead < size) {
                int read = is.read(b, totalRead, size - totalRead);
                if (read == -1) break;
                totalRead += read;
            }
            return b;
        } finally {
            is.close();
        }
    }

    public void checkAndPrepareModel(ModelStatusCallback callback) {
        if (modelReady) {
            // Model is ready
            callback.onStatusChecked(1); // available
            callback.onModelReady();
        } else if (isLoading) {
            // Model is still loading, store callback to notify when done
            pendingCallback = callback;
            callback.onStatusChecked(0); // loading
            Log.d(TAG, "Model is loading, callback will be notified when ready");
        } else {
            // Model failed to load or not started
            callback.onStatusChecked(2); // unavailable
            callback.onDownloadFailed("ONNX model not loaded");
        }
    }

    /**
     * Main entry: send SMS text here.
     * Returns JSON with type, amount, description (null if confidence low).
     */
    public void generateContent(String smsText, ContentGenerationCallback callback) {
        // Safety check: Never process if model is not ready
        if (!modelReady) {
            callback.onFailure("Model still loading, try again");
            return;
        }
        
        Executor executor = ContextCompat.getMainExecutor(context);
        executor.execute(() -> {
            try {
                // Double-check model is ready
                if (session == null || inputVectorSize <= 0) {
                    callback.onFailure("Model sessions not initialized");
                    return;
                }

                // Preprocess SMS text into float vector
                float[] features = preprocessSmsText(smsText, inputVectorSize);

                // Prepare tensor: shape [1, inputVectorSize] for batch inference
                long[] shape = new long[]{1, inputVectorSize};
                FloatBuffer fb = FloatBuffer.wrap(features);
                
                // Get input name
                Set<String> inputNames = session.getInputNames();
                String inputName = null;
                if (!inputNames.isEmpty()) {
                    inputName = inputNames.iterator().next();
                }
                if (inputName == null) {
                    callback.onFailure("Model has no input");
                    return;
                }

                // Run inference
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, fb, shape);
                     OrtSession.Result result = session.run(java.util.Collections.singletonMap(inputName, inputTensor))) {
                    
                    // Parse model output
                    ModelOutput output = parseModelOutput(result, smsText);
                    
                    // Build JSON response
                    String json = buildJsonResponse(output);
                    callback.onSuccess(json);
                }

            } catch (Exception e) {
                Log.e(TAG, "ONNX inference error", e);
                callback.onFailure("Inference failed: " + e.getMessage());
            }
        });
    }

    /**
     * Preprocess SMS text into a float vector of the required size.
     * Pads with 0.0f if shorter than required size.
     */
    private float[] preprocessSmsText(String sms, int targetSize) {
        if (sms == null) {
            sms = "";
        }
        
        // Convert SMS to lowercase for consistent processing
        String normalized = sms.toLowerCase().trim();
        
        // Convert characters to float values (simple character encoding)
        // Each character is encoded as its ASCII value normalized to [0, 1]
        float[] features = new float[targetSize];
        int smsLength = normalized.length();
        
        for (int i = 0; i < targetSize; i++) {
            if (i < smsLength) {
                char c = normalized.charAt(i);
                // Normalize ASCII value to [0, 1] range
                features[i] = (float) c / 128.0f;
            } else {
                // Pad with zeros
                features[i] = 0.0f;
            }
        }
        
        return features;
    }

    /**
     * Parse model output to extract type, amount, description.
     */
    private ModelOutput parseModelOutput(OrtSession.Result result, String smsText) throws OrtException {
        ModelOutput output = new ModelOutput();
        
        // Get first output (most common case)
        Object outputValue = result.get(0).getValue();
        
        // Handle different output formats
        float[] probabilities = extractProbabilitiesFromResult(outputValue);
        
        if (probabilities.length == 0) {
            Log.w(TAG, "Empty model output");
            return output; // All nulls
        }
        
        // Find max probability and its index
        int maxIdx = argMax(probabilities);
        float maxProb = probabilities[maxIdx];
        
        // Check confidence threshold
        if (maxProb < CONFIDENCE_THRESHOLD) {
            Log.d(TAG, "Low confidence: " + maxProb + " < " + CONFIDENCE_THRESHOLD);
            return output; // All nulls due to low confidence
        }
        
        // Map index to type
        // Assuming: 0 = debit, 1 = credit, 2 = none/other
        if (maxIdx == 0) {
            output.type = "debit";
        } else if (maxIdx == 1) {
            output.type = "credit";
        } else {
            // Index 2 or higher = none/unknown
            output.type = null;
            return output; // Return nulls if type not detected
        }
        
        // Extract amount and description from SMS text using heuristics
        // (Model might only output type probabilities)
        output.amount = extractAmountHeuristic(smsText);
        output.description = extractDescriptionHeuristic(smsText);
        
        return output;
    }

    /**
     * Build JSON response from model output.
     */
    private String buildJsonResponse(ModelOutput output) {
        StringBuilder json = new StringBuilder("{");
        
        // Type
        json.append("\"type\":");
        if (output.type == null) {
            json.append("null");
        } else {
            json.append("\"").append(escapeJson(output.type)).append("\"");
        }
        
        // Amount
        json.append(",\"amount\":");
        if (output.amount == null) {
            json.append("null");
        } else {
            json.append(output.amount);
        }
        
        // Description
        json.append(",\"description\":");
        if (output.description == null) {
            json.append("null");
        } else {
            json.append("\"").append(escapeJson(output.description)).append("\"");
        }
        
        json.append("}");
        return json.toString();
    }

    /**
     * Extract probabilities from model output (handles various formats).
     */
    private float[] extractProbabilitiesFromResult(Object outputValue) {
        try {
            if (outputValue instanceof float[][]) {
                float[][] arr = (float[][]) outputValue;
                return arr.length > 0 ? arr[0] : new float[0];
            } else if (outputValue instanceof double[][]) {
                double[][] arr = (double[][]) outputValue;
                if (arr.length == 0) return new float[0];
                double[] row = arr[0];
                float[] out = new float[row.length];
                for (int i = 0; i < row.length; i++) {
                    out[i] = (float) row[i];
                }
                return out;
            } else if (outputValue instanceof float[]) {
                return (float[]) outputValue;
            } else if (outputValue instanceof double[]) {
                double[] a = (double[]) outputValue;
                float[] out = new float[a.length];
                for (int i = 0; i < a.length; i++) {
                    out[i] = (float) a[i];
                }
                return out;
            } else if (outputValue instanceof Number) {
                return new float[]{((Number) outputValue).floatValue()};
            } else {
                Log.w(TAG, "Unexpected output type: " + outputValue.getClass().getName());
                return new float[0];
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to parse model output", e);
            return new float[0];
        }
    }

    private int argMax(float[] arr) {
        if (arr.length == 0) return -1;
        int idx = 0;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    private String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * Extract amount from SMS text using heuristics.
     */
    private Double extractAmountHeuristic(String sms) {
        if (sms == null || sms.isEmpty()) {
            return null;
        }
        
        // Remove commas and find first number-like substring with optional decimal
        String cleaned = sms.replaceAll(",", "");
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(\\d+\\.?\\d{0,2})");
        java.util.regex.Matcher matcher = pattern.matcher(cleaned);
        
        if (matcher.find()) {
            try {
                return Double.parseDouble(matcher.group(1));
            } catch (NumberFormatException e) {
                Log.w(TAG, "Failed to parse amount: " + matcher.group(1));
            }
        }
        
        return null;
    }

    /**
     * Extract description from SMS text using heuristics.
     */
    private String extractDescriptionHeuristic(String sms) {
        if (sms == null || sms.isEmpty()) {
            return null;
        }
        
        String lower = sms.toLowerCase();
        
        // Try to extract text after "to"
        if (lower.contains(" to ")) {
            int idx = lower.indexOf(" to ");
            String desc = sms.substring(idx + 4).trim();
            // Remove amount if present
            desc = desc.replaceAll("\\d+\\.?\\d*", "").trim();
            if (!desc.isEmpty()) {
                return desc.length() > 50 ? desc.substring(0, 50) : desc;
            }
        }
        
        // Try to extract text before "via"
        if (lower.contains(" via ")) {
            int idx = lower.indexOf(" via ");
            String desc = sms.substring(0, idx).trim();
            if (!desc.isEmpty()) {
                return desc.length() > 50 ? desc.substring(0, 50) : desc;
            }
        }
        
        // Fallback: return first meaningful part
        String trimmed = sms.trim();
        // Remove common prefixes
        trimmed = trimmed.replaceFirst("(?i)^.*?(?:debited|credited|received|spent)\\s*", "");
        trimmed = trimmed.trim();
        
        if (trimmed.length() > 50) {
            trimmed = trimmed.substring(0, 50);
        }
        
        return trimmed.isEmpty() ? null : trimmed;
    }

    /**
     * Internal class to hold model output.
     */
    private static class ModelOutput {
        String type = null;
        Double amount = null;
        String description = null;
    }

    public void warmup() {
        // Optional: Run a dummy inference to warm up the model
        if (modelReady && inputVectorSize > 0) {
            try {
                float[] dummy = new float[inputVectorSize];
                long[] shape = new long[]{1, inputVectorSize};
                FloatBuffer fb = FloatBuffer.wrap(dummy);
                Set<String> inputNames = session.getInputNames();
                if (!inputNames.isEmpty()) {
                    String inputName = inputNames.iterator().next();
                    try (OnnxTensor tensor = OnnxTensor.createTensor(env, fb, shape);
                         OrtSession.Result result = session.run(java.util.Collections.singletonMap(inputName, tensor))) {
                        Log.d(TAG, "Model warmed up");
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "Warmup failed", e);
            }
        }
    }

    public boolean isModelReady() {
        return modelReady;
    }
}
