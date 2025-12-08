package com.example.expensetracker;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Helper class for SMS transaction parsing using ONNX Runtime Mobile.
 * Relies ONLY on the ONNX model - no manual/heuristic parsing.
 * Works fully offline without any Google Play Services dependencies.
 */
public class GenerativeModelHelper {
    private static final String TAG = "GenerativeModelHelper";
    
    // Feature status constants
    private static final int FEATURE_STATUS_AVAILABLE = 1;
    private static final int FEATURE_STATUS_UNAVAILABLE = 2;
    private static final int FEATURE_STATUS_DOWNLOADING = 3;
    private static final int FEATURE_STATUS_DOWNLOADABLE = 4;
    
    // Model file path
    private static final String MODEL_FILE = "model.onnx";
    
    private Context context;
    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private boolean modelReady = false;
    private ExecutorService executorService;
    
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
        this.executorService = Executors.newSingleThreadExecutor();
        initializeModel();
    }
    
    /**
     * Initialize ONNX Runtime model from assets
     */
    private void initializeModel() {
        executorService.execute(() -> {
            try {
                // Initialize ONNX Runtime environment
                ortEnvironment = OrtEnvironment.getEnvironment();
                
                // Load model from assets
                AssetManager assetManager = context.getAssets();
                InputStream inputStream = assetManager.open(MODEL_FILE);
                
                // Read model bytes
                byte[] modelBytes = new byte[inputStream.available()];
                inputStream.read(modelBytes);
                inputStream.close();
                
                // Create ONNX session
                OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                sessionOptions.setIntraOpNumThreads(2);
                
                ortSession = ortEnvironment.createSession(modelBytes, sessionOptions);
                
                modelReady = true;
                Log.d(TAG, "ONNX model loaded successfully");
                
            } catch (Exception e) {
                Log.e(TAG, "Failed to load ONNX model", e);
                modelReady = false;
            }
        });
    }
    
    /**
     * Check and prepare model (maintains API compatibility)
     */
    public void checkAndPrepareModel(ModelStatusCallback callback) {
        checkAICoreAvailability(callback);
    }
    
    /**
     * Check model availability (maintains API compatibility)
     */
    public void checkAICoreAvailability(ModelStatusCallback callback) {
        executorService.execute(() -> {
            android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
            
            // Wait a bit for model initialization
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            mainHandler.post(() -> {
                callback.onStatusChecked(
                    modelReady ? FEATURE_STATUS_AVAILABLE : FEATURE_STATUS_UNAVAILABLE
                );
                if (modelReady) {
                    callback.onModelReady();
                } else {
                    callback.onDownloadFailed("ONNX model failed to load. Please ensure model.onnx exists in assets folder.");
                }
            });
        });
    }
    
    /**
     * Generate content from text-only input
     */
    public void generateContent(String prompt, ContentGenerationCallback callback) {
        generateContent(prompt, null, callback);
    }
    
    /**
     * Generate content from multimodal input (image + text)
     * Note: Image processing not implemented, uses text only
     */
    public void generateContent(String textPrompt, Bitmap image, ContentGenerationCallback callback) {
        // Extract SMS text from prompt if it contains "SMS:"
        String smsText = textPrompt;
        if (textPrompt.contains("SMS:")) {
            int smsStart = textPrompt.indexOf("SMS:") + 4;
            int smsEnd = textPrompt.length();
            if (textPrompt.indexOf("\"", smsStart) != -1) {
                smsStart = textPrompt.indexOf("\"", smsStart) + 1;
                smsEnd = textPrompt.indexOf("\"", smsStart);
                if (smsEnd == -1) smsEnd = textPrompt.length();
            }
            smsText = textPrompt.substring(smsStart, smsEnd).trim();
        }
        
        // Analyze SMS using ONLY ONNX model
        analyzeSMS(smsText, callback);
    }
    
    /**
     * Analyze SMS message using ONLY ONNX Runtime model
     */
    private void analyzeSMS(String smsText, ContentGenerationCallback callback) {
        if (!modelReady) {
            callback.onFailure("Model not ready. ONNX model failed to initialize.");
            return;
        }
        
        if (ortSession == null) {
            callback.onFailure("ONNX session not available. Model initialization failed.");
            return;
        }
        
        if (smsText == null || smsText.trim().isEmpty()) {
            callback.onFailure("SMS text is empty");
            return;
        }
        
        executorService.execute(() -> {
            try {
                // Use ONLY ONNX model - no fallback
                TransactionResult result = parseSMSWithONNX(smsText);
                String jsonResponse = result.toJson();
                
                android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
                mainHandler.post(() -> callback.onSuccess(jsonResponse));
                
            } catch (Exception e) {
                Log.e(TAG, "ONNX inference failed", e);
                android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
                mainHandler.post(() -> callback.onFailure("ONNX model inference failed: " + e.getMessage() + 
                    ". Please ensure your model.onnx is properly trained and matches the expected input/output format."));
            }
        });
    }
    
    /**
     * Parse SMS using ONLY ONNX Runtime inference
     * Model must handle: type classification, amount extraction, description generation, fraud detection
     */
    private TransactionResult parseSMSWithONNX(String smsText) throws Exception {
        // Prepare input tensor
        float[] inputFeatures = preprocessText(smsText);
        
        // Create input tensor
        long[] shape = {1, inputFeatures.length};
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputFeatures), shape);
        
        // Prepare inputs map
        Map<String, OnnxTensor> inputs = new HashMap<>();
        // Adjust input name based on your model - common names: "input", "text", "features"
        inputs.put("input", inputTensor);
        
        // Run inference - model does ALL the work
        OrtSession.Result outputs = ortSession.run(inputs);
        
        // Extract outputs - model returns everything we need
        // Adjust output indices/names based on your model structure
        // Expected outputs: [type, amount, description, fraud]
        OnnxTensor typeTensor = (OnnxTensor) outputs.get(0);
        OnnxTensor amountTensor = (OnnxTensor) outputs.get(1);
        OnnxTensor descriptionTensor = (OnnxTensor) outputs.get(2);
        OnnxTensor fraudTensor = (OnnxTensor) outputs.get(3);
        
        // Parse outputs - model provides all values
        String type = parseTypeOutput(typeTensor);
        Double amount = parseAmountOutput(amountTensor);
        String description = parseDescriptionOutput(descriptionTensor);
        boolean fraud = parseFraudOutput(fraudTensor);
        
        // Cleanup
        inputTensor.close();
        typeTensor.close();
        amountTensor.close();
        descriptionTensor.close();
        fraudTensor.close();
        outputs.close();
        
        // If fraud detected, return null values as specified
        if (fraud) {
            return new TransactionResult(null, null, null, true);
        }
        
        return new TransactionResult(type, amount, description, false);
    }
    
    /**
     * Preprocess text for ONNX model input
     * Minimal preprocessing - model handles all computation
     * This only converts text to raw bytes/characters for model input
     */
    private float[] preprocessText(String text) {
        // Minimal preprocessing - just convert text to numerical representation
        // Model handles ALL computation - no manual feature extraction
        byte[] textBytes = text.getBytes();
        float[] features = new float[textBytes.length];
        
        // Simple byte-to-float conversion - model does all the work
        for (int i = 0; i < textBytes.length; i++) {
            features[i] = (textBytes[i] & 0xFF) / 255.0f; // Normalize to 0-1
        }
        
        // Pad or truncate to model's expected input size (adjust 128 to your model's input size)
        float[] modelInput = new float[128];
        int copyLength = Math.min(features.length, modelInput.length);
        System.arraycopy(features, 0, modelInput, 0, copyLength);
        
        return modelInput;
    }
    
    /**
     * Parse type output from ONNX tensor
     * Model returns: [credit_probability, debit_probability]
     * NO manual computation - just extract model's decision
     */
    private String parseTypeOutput(OnnxTensor tensor) {
        try {
            float[][] output = (float[][]) tensor.getValue();
            if (output.length > 0 && output[0].length >= 2) {
                // Model already computed probabilities - just extract the decision
                // Model output: [credit_prob, debit_prob]
                float creditProb = output[0][0];
                float debitProb = output[0][1];
                
                // Return model's decision - no manual thresholding or computation
                if (creditProb > debitProb) {
                    return "credit";
                } else if (debitProb > creditProb) {
                    return "debit";
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Error parsing type output", e);
        }
        return null;
    }
    
    /**
     * Parse amount output from ONNX tensor
     * Model returns: [amount_value]
     * NO manual computation - model already extracted the amount
     */
    private Double parseAmountOutput(OnnxTensor tensor) {
        try {
            float[][] output = (float[][]) tensor.getValue();
            if (output.length > 0 && output[0].length > 0) {
                // Model already computed the amount - just return it
                double amount = output[0][0];
                return amount; // Return model's output directly - no validation or filtering
            }
        } catch (Exception e) {
            Log.w(TAG, "Error parsing amount output", e);
        }
        return null;
    }
    
    /**
     * Parse description output from ONNX tensor
     * Model returns: description text or token IDs
     * NO manual computation - model already generated the description
     */
    private String parseDescriptionOutput(OnnxTensor tensor) {
        try {
            // Model already generated the description - just extract it
            // Option 1: Direct text output from model
            String[][] output = (String[][]) tensor.getValue();
            if (output.length > 0 && output[0].length > 0) {
                return output[0][0]; // Return model's generated description directly
            }
            
            // Option 2: If model outputs token IDs, you may need a tokenizer
            // But this should be minimal - model did the generation work
            // int[][] tokenIds = (int[][]) tensor.getValue();
            // return decodeTokens(tokenIds); // Only decoding, no manual generation
            
        } catch (Exception e) {
            Log.w(TAG, "Error parsing description output", e);
        }
        return null;
    }
    
    /**
     * Parse fraud output from ONNX tensor
     * Model returns: [fraud_probability] or [fraud_class]
     * NO manual computation - model already detected fraud
     */
    private boolean parseFraudOutput(OnnxTensor tensor) {
        try {
            float[][] output = (float[][]) tensor.getValue();
            if (output.length > 0 && output[0].length > 0) {
                // Model already computed fraud detection - just extract the result
                // If model outputs probability, use it directly
                // If model outputs class (0/1), use it directly
                float fraudValue = output[0][0];
                return fraudValue > 0.5f; // Minimal threshold - model did the detection work
            }
        } catch (Exception e) {
            Log.w(TAG, "Error parsing fraud output", e);
        }
        return false;
    }
    
    /**
     * Warm up the model for faster first inference
     */
    public void warmup() {
        if (ortSession != null && modelReady) {
            executorService.execute(() -> {
                try {
                    // Run a dummy inference to warm up
                    float[] dummyInput = new float[128]; // Adjust size
                    long[] shape = {1, 128};
                    OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(dummyInput), shape);
                    Map<String, OnnxTensor> inputs = new HashMap<>();
                    inputs.put("input", inputTensor);
                    OrtSession.Result outputs = ortSession.run(inputs);
                    outputs.close();
                    inputTensor.close();
                } catch (Exception e) {
                    Log.w(TAG, "Warmup failed", e);
                }
            });
        }
    }
    
    /**
     * Check if model is ready to use
     */
    public boolean isModelReady() {
        return modelReady && ortSession != null;
    }
    
    /**
     * Cleanup resources
     */
    public void close() {
        if (ortSession != null) {
            try {
                ortSession.close();
            } catch (Exception e) {
                Log.e(TAG, "Error closing ONNX session", e);
            }
            ortSession = null;
        }
        if (executorService != null) {
            executorService.shutdown();
        }
    }
    
    /**
     * Inner class to hold transaction parsing results
     */
    private static class TransactionResult {
        String type;
        Double amount;
        String description;
        boolean fraud;
        
        TransactionResult(String type, Double amount, String description, boolean fraud) {
            this.type = type;
            this.amount = amount;
            this.description = description;
            this.fraud = fraud;
        }
        
        String toJson() {
            StringBuilder json = new StringBuilder();
            json.append("{\n");
            json.append("  \"type\": ");
            if (type == null) {
                json.append("null");
            } else {
                json.append("\"").append(type).append("\"");
            }
            json.append(",\n");
            json.append("  \"amount\": ");
            if (amount == null) {
                json.append("null");
            } else {
                json.append(amount.intValue()); // Return as number
            }
            json.append(",\n");
            json.append("  \"description\": ");
            if (description == null) {
                json.append("null");
            } else {
                json.append("\"").append(escapeJson(description)).append("\"");
            }
            json.append(",\n");
            json.append("  \"fraud\": ").append(fraud).append("\n");
            json.append("}");
            return json.toString();
        }
        
        private String escapeJson(String str) {
            return str.replace("\\", "\\\\")
                     .replace("\"", "\\\"")
                     .replace("\n", "\\n")
                     .replace("\r", "\\r")
                     .replace("\t", "\\t");
        }
    }
}