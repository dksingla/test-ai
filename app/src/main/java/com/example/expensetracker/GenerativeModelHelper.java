package com.example.expensetracker;

import android.content.Context;
import android.util.Log;

import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX helper: loads two ONNX models (type + fraud) and runs inference.
 *
 * Requirement: assets/model_type.onnx and assets/model_fraud.onnx
 */
public class GenerativeModelHelper {
    private static final String TAG = "GenerativeModelHelper";

    private final Context context;
    private OrtEnvironment env;
    private OrtSession sessionType;
    private OrtSession sessionFraud;
    private volatile boolean modelReady = false;
    private volatile boolean isLoading = false;
    private ModelStatusCallback pendingCallback = null;

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

                byte[] modelTypeBytes = loadAssetBytes("model_type.onnx");
                sessionType = env.createSession(modelTypeBytes);

                byte[] modelFraudBytes = loadAssetBytes("model_fraud.onnx");
                sessionFraud = env.createSession(modelFraudBytes);

                modelReady = true;
                isLoading = false;
                Log.d(TAG, "✅ ONNX models loaded");
                
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
                Log.e(TAG, "Failed to load ONNX models", e);
                modelReady = false;
                isLoading = false;
                
                // Notify pending callback of failure on main thread
                if (pendingCallback != null) {
                    ModelStatusCallback callback = pendingCallback;
                    pendingCallback = null;
                    mainExecutor.execute(() -> {
                        callback.onStatusChecked(2); // unavailable
                        callback.onDownloadFailed("Failed to load ONNX models: " + e.getMessage());
                    });
                }
            }
        });
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
            // Models are ready
            callback.onStatusChecked(1); // available
            callback.onModelReady();
        } else if (isLoading) {
            // Models are still loading, store callback to notify when done
            pendingCallback = callback;
            callback.onStatusChecked(0); // loading
            Log.d(TAG, "Models are loading, callback will be notified when ready");
        } else {
            // Models failed to load or not started
            callback.onStatusChecked(2); // unavailable
            callback.onDownloadFailed("ONNX models not loaded");
        }
    }

    /**
     * Main entry: send SMS text here.
     * If model predicts 'none' type => returns nulls + fraud=true.
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
                // Compute features exactly like training script
                float[] features = computeFeatures(smsText); // length = 6

                // Double-check models are ready (defensive programming)
                if (sessionType == null || sessionFraud == null) {
                    callback.onFailure("Model sessions not initialized");
                    return;
                }

                // Prepare tensor: shape [1, n_features]
                long[] shape = new long[]{1, features.length};
                FloatBuffer fb = FloatBuffer.wrap(features);
                
                // RUN type model
                String typePred = "none";
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, fb, shape);
                     OrtSession.Result r = sessionType.run(java.util.Collections.singletonMap(
                        sessionType.getInputNames().iterator().next(), inputTensor))) {

                    // Most sklearn->ONNX exports produce a probability output named 'label' or array. We attempt to read first output
                    // Get first output as float array/probs
                    Object out0 = r.get(0).getValue();
                    // out0 may be probability array shape [1,3] -> we handle common cases
                    float[] probs = extractProbabilitiesFromResult(out0);
                    int argmax = argMax(probs);
                    if (argmax == 0) typePred = "debit";
                    else if (argmax == 1) typePred = "credit";
                    else typePred = "none";
                }

                // RUN fraud model -> get probability
                float fraudProb = 0.0f;
                fb.rewind(); // Reset buffer position for reuse
                try (OnnxTensor inputTensor2 = OnnxTensor.createTensor(env, fb, shape);
                     OrtSession.Result r2 = sessionFraud.run(java.util.Collections.singletonMap(
                        sessionFraud.getInputNames().iterator().next(), inputTensor2))) {
                    Object out0 = r2.get(0).getValue();
                    float[] probs = extractProbabilitiesFromResult(out0);
                    // fraud model was trained binary; pick probability of class 1 if provided, or single-score
                    if (probs.length == 1) {
                        fraudProb = probs[0];
                    } else if (probs.length >= 2) {
                        fraudProb = probs[1];
                    }
                }

                // If type == none -> your rule: return nulls and fraud=true
                if ("none".equals(typePred)) {
                    String json = "{\"type\":null,\"amount\":null,\"description\":null,\"fraud\":true}";
                    callback.onSuccess(json);
                    return;
                }

                // Otherwise compute amount & description heuristics
                Double amount = extractAmountHeuristic(smsText);
                String desc = extractDescriptionHeuristic(smsText);

                boolean fraud = fraudProb >= 0.5f; // threshold
                String json = "{"
                        + "\"type\":\"" + typePred + "\","
                        + "\"amount\":" + (amount == null ? "null" : amount) + ","
                        + "\"description\":" + (desc == null ? "null" : "\"" + escapeJson(desc) + "\"") + ","
                        + "\"fraud\":" + fraud
                        + "}";
                callback.onSuccess(json);

            } catch (Exception e) {
                Log.e(TAG, "ONNX inference error, using fallback", e);
                callback.onFailure("Inference failed: " + e.getMessage());
            }
        });
    }

    // -- Helpers --

    private float[] computeFeatures(String sms) {
        String s = sms == null ? "" : sms.toLowerCase();
        float f0 = (s.contains("debit") || s.contains("debited") || s.contains("spent")) ? 1f : 0f;
        float f1 = (s.contains("credit") || s.contains("credited") || s.contains("received")) ? 1f : 0f;
        float f2 = s.contains("to ") ? 1f : 0f;
        float f3 = s.contains("from ") ? 1f : 0f;
        float f4 = countNumbers(s);
        float f5 = Math.min((float)s.length(), 10000f);
        return new float[]{f0, f1, f2, f3, f4, f5};
    }

    private int countNumbers(String s) {
        int count = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (Character.isDigit(s.charAt(i))) {
                count++;
                // skip sequence of digits
                while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) i++;
            }
        }
        return count;
    }

    private float[] extractProbabilitiesFromResult(Object out0) {
        // Possible shapes:
        // - float[][] (1 x N) -> return flatten
        // - double[][] -> cast
        // - float[] -> return
        // - double[] -> cast
        try {
            if (out0 instanceof float[][]) {
                float[][] arr = (float[][]) out0;
                return arr[0];
            } else if (out0 instanceof double[][]) {
                double[][] arr = (double[][]) out0;
                double[] row = arr[0];
                float[] out = new float[row.length];
                for (int i = 0; i < row.length; i++) out[i] = (float) row[i];
                return out;
            } else if (out0 instanceof float[]) {
                return (float[]) out0;
            } else if (out0 instanceof double[]) {
                double[] a = (double[]) out0;
                float[] out = new float[a.length];
                for (int i = 0; i < a.length; i++) out[i] = (float) a[i];
                return out;
            } else if (out0 instanceof java.lang.Number) {
                return new float[]{((Number) out0).floatValue()};
            } else {
                // fallback: try toString parsing
                String s = out0.toString();
                // attempt simple parse of bracketed numbers
                s = s.replaceAll("[\\[\\]]", "");
                String[] parts = s.split("[,\\s]+");
                float[] out = new float[parts.length];
                for (int i = 0; i < parts.length; ++i) {
                    try { out[i] = Float.parseFloat(parts[i]); } catch (Exception ex) { out[i] = 0f; }
                }
                return out;
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to parse model output", e);
            return new float[]{0f};
        }
    }

    private int argMax(float[] arr) {
        int idx = 0;
        float m = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] > m) { m = arr[i]; idx = i; }
        }
        return idx;
    }

    private Double extractAmountHeuristic(String sms) {
        // Find first number-like substring with optional decimal and thousand separators
        String s = sms.replaceAll(",", "");
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d+\\.?\\d{0,2})").matcher(s);
        if (m.find()) {
            try {
                return Double.parseDouble(m.group(1));
            } catch (Exception e) { return null; }
        }
        return null;
    }

    private String extractDescriptionHeuristic(String sms) {
        String lower = sms.toLowerCase();
        if (lower.contains(" to ")) {
            int idx = lower.indexOf(" to ");
            return sms.substring(idx + 4).trim();
        }
        if (lower.contains(" via ")) {
            int idx = lower.indexOf(" via ");
            return sms.substring(0, idx).trim();
        }
        // fallback: return first 30 chars
        String trimmed = sms.trim();
        return trimmed.length() > 30 ? trimmed.substring(0,30) : trimmed;
    }

    private String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    // Very simple fallback using heuristics if models unavailable
    private String fallbackResponseBasedOnHeuristics(String sms) {
        String lower = sms == null ? "" : sms.toLowerCase();
        String type = null;
        if (lower.contains("debit") || lower.contains("debited") || lower.contains("spent")) type = "debit";
        if (lower.contains("credit") || lower.contains("credited") || lower.contains("received")) type = "credit";
        Double amount = extractAmountHeuristic(sms);
        String desc = extractDescriptionHeuristic(sms);
        boolean fraud = lower.contains("otp") || lower.contains("blocked");
        if (type == null) {
            // rule: if type not found => nulls + fraud true
            return "{\"type\":null,\"amount\":null,\"description\":null,\"fraud\":true}";
        }
        return "{"
                + "\"type\":\"" + type + "\","
                + "\"amount\":" + (amount == null ? "null" : amount) + ","
                + "\"description\":" + (desc == null ? "null" : "\"" + escapeJson(desc) + "\"") + ","
                + "\"fraud\":" + fraud
                + "}";
    }

    public void warmup() {
        // no-op
    }

    public boolean isModelReady() { return modelReady; }
}
