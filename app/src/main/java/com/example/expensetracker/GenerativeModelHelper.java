package com.example.expensetracker;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helper class for SMS transaction parsing using ONNX Runtime Mobile.
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
    private boolean useHeuristics = false;
    private ExecutorService executorService;
    
    // Fraud detection keywords
    private static final String[] FRAUD_INDICATORS = {
        "urgent", "verify", "suspended", "click here", "link", "password",
        "account locked", "verify now", "immediately", "act now", "limited time",
        "prize", "winner", "congratulations", "free money", "claim now",
        "phishing", "scam", "suspicious", "verify account", "update now"
    };
    
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
                useHeuristics = false;
                Log.d(TAG, "ONNX model loaded successfully");
                
            } catch (Exception e) {
                Log.w(TAG, "Failed to load ONNX model, using heuristic fallback", e);
                useHeuristics = true;
                modelReady = true; // Still ready with heuristics
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
                    callback.onDownloadFailed("Model initialization failed");
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
        
        // Analyze SMS
        analyzeSMS(smsText, callback);
    }
    
    /**
     * Analyze SMS message using ONNX Runtime or heuristic fallback
     */
    private void analyzeSMS(String smsText, ContentGenerationCallback callback) {
        if (!modelReady) {
            callback.onFailure("Model not ready");
            return;
        }
        
        if (smsText == null || smsText.trim().isEmpty()) {
            callback.onFailure("SMS text is empty");
            return;
        }
        
        executorService.execute(() -> {
            try {
                TransactionResult result;
                
                if (useHeuristics || ortSession == null) {
                    result = parseSMSWithHeuristics(smsText);
                } else {
                    result = parseSMSWithONNX(smsText);
                }
                
                String jsonResponse = result.toJson();
                
                android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
                mainHandler.post(() -> callback.onSuccess(jsonResponse));
                
            } catch (Exception e) {
                Log.e(TAG, "SMS analysis failed", e);
                android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
                mainHandler.post(() -> callback.onFailure("Analysis failed: " + e.getMessage()));
            }
        });
    }
    
    /**
     * Parse SMS using ONNX Runtime inference
     */
    private TransactionResult parseSMSWithONNX(String smsText) {
        try {
            // Check for fraud first
            boolean isFraud = detectFraud(smsText);
            if (isFraud) {
                return new TransactionResult(null, null, null, true);
            }
            
            // Prepare input tensor
            // Assuming model expects text embedding or tokenized input
            // This is a simplified example - adjust based on your actual model input format
            float[] inputFeatures = preprocessText(smsText);
            
            // Create input tensor
            long[] shape = {1, inputFeatures.length};
            OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputFeatures), shape);
            
            // Prepare inputs map
            Map<String, OnnxTensor> inputs = new HashMap<>();
            // Adjust input name based on your model - common names: "input", "text", "features"
            inputs.put("input", inputTensor);
            
            // Run inference
            OrtSession.Result outputs = ortSession.run(inputs);
            
            // Extract outputs
            // Adjust output names based on your model - common names: "output", "type", "amount", "description"
            OnnxTensor typeTensor = (OnnxTensor) outputs.get(0);
            OnnxTensor amountTensor = (OnnxTensor) outputs.get(1);
            OnnxTensor descriptionTensor = (OnnxTensor) outputs.get(2);
            
            // Parse outputs
            String type = parseTypeOutput(typeTensor);
            Double amount = parseAmountOutput(amountTensor);
            String description = parseDescriptionOutput(descriptionTensor);
            
            // Cleanup
            inputTensor.close();
            typeTensor.close();
            amountTensor.close();
            descriptionTensor.close();
            outputs.close();
            
            return new TransactionResult(type, amount, description, false);
            
        } catch (Exception e) {
            Log.w(TAG, "ONNX inference failed, falling back to heuristics", e);
            return parseSMSWithHeuristics(smsText);
        }
    }
    
    /**
     * Preprocess text for ONNX model input
     * Adjust this based on your model's expected input format
     */
    private float[] preprocessText(String text) {
        // Simple text embedding - replace with your model's preprocessing
        // This is a placeholder - you may need tokenization, word embeddings, etc.
        String lowerText = text.toLowerCase();
        float[] features = new float[128]; // Adjust size based on your model
        
        // Simple feature extraction
        String[] words = lowerText.split("\\s+");
        for (int i = 0; i < Math.min(words.length, features.length); i++) {
            features[i] = words[i].hashCode() % 1000 / 1000.0f;
        }
        
        return features;
    }
    
    /**
     * Parse type output from ONNX tensor
     */
    private String parseTypeOutput(OnnxTensor tensor) {
        try {
            float[][] output = (float[][]) tensor.getValue();
            if (output.length > 0 && output[0].length >= 2) {
                // Assuming binary classification: [credit_prob, debit_prob]
                if (output[0][0] > output[0][1] && output[0][0] > 0.5f) {
                    return "credit";
                } else if (output[0][1] > output[0][0] && output[0][1] > 0.5f) {
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
     */
    private Double parseAmountOutput(OnnxTensor tensor) {
        try {
            float[][] output = (float[][]) tensor.getValue();
            if (output.length > 0 && output[0].length > 0) {
                double amount = output[0][0];
                return amount > 0 ? amount : null;
            }
        } catch (Exception e) {
            Log.w(TAG, "Error parsing amount output", e);
        }
        return null;
    }
    
    /**
     * Parse description output from ONNX tensor
     */
    private String parseDescriptionOutput(OnnxTensor tensor) {
        try {
            // Adjust based on your model's output format
            // This is a placeholder - you may need to decode from token IDs
            String[][] output = (String[][]) tensor.getValue();
            if (output.length > 0 && output[0].length > 0) {
                return output[0][0];
            }
        } catch (Exception e) {
            Log.w(TAG, "Error parsing description output", e);
        }
        return null;
    }
    
    /**
     * Parse SMS using heuristic fallback
     */
    private TransactionResult parseSMSWithHeuristics(String smsText) {
        // Check for fraud
        boolean isFraud = detectFraud(smsText);
        if (isFraud) {
            return new TransactionResult(null, null, null, true);
        }
        
        // Determine transaction type
        String type = determineTransactionType(smsText);
        
        // Extract amount
        Double amount = extractAmount(smsText);
        
        // Generate description
        String description = generateDescription(smsText, type);
        
        return new TransactionResult(type, amount, description, false);
    }
    
    /**
     * Detect fraud indicators in SMS
     */
    private boolean detectFraud(String smsText) {
        String lowerText = smsText.toLowerCase();
        for (String indicator : FRAUD_INDICATORS) {
            if (lowerText.contains(indicator.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Determine transaction type using heuristics
     */
    private String determineTransactionType(String smsText) {
        String lowerText = smsText.toLowerCase();
        
        String[] creditKeywords = {
            "credited", "credit", "received", "deposit", "income", "salary",
            "refund", "cashback", "reward", "bonus", "incoming", "added"
        };
        
        String[] debitKeywords = {
            "debited", "debit", "paid", "payment", "purchase", "withdrawal",
            "transfer", "sent", "outgoing", "spent", "expense", "deducted"
        };
        
        int creditScore = 0;
        int debitScore = 0;
        
        for (String keyword : creditKeywords) {
            if (lowerText.contains(keyword)) {
                creditScore++;
            }
        }
        
        for (String keyword : debitKeywords) {
            if (lowerText.contains(keyword)) {
                debitScore++;
            }
        }
        
        if (creditScore > debitScore && creditScore > 0) {
            return "credit";
        } else if (debitScore > creditScore && debitScore > 0) {
            return "debit";
        }
        
        return null;
    }
    
    /**
     * Extract transaction amount from SMS
     */
    private Double extractAmount(String text) {
        String[] patterns = {
            "(?:rs\\.?|rupees?|inr)\\s*:?\\s*(\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d{2})?)",
            "₹\\s*:?\\s*(\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d{2})?)",
            "(\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d{2})?)\\s*(?:rs\\.?|rupees?|inr)",
            "amount[\\s:]+(\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d{2})?)",
            "\\b(\\d{1,3}(?:,\\d{2,3})*(?:\\.\\d{2})?)\\b"
        };
        
        for (String patternStr : patterns) {
            Pattern pattern = Pattern.compile(patternStr, Pattern.CASE_INSENSITIVE);
            Matcher matcher = pattern.matcher(text);
            if (matcher.find()) {
                try {
                    String amountStr = matcher.group(1).replace(",", "");
                    return Double.parseDouble(amountStr);
                } catch (NumberFormatException e) {
                    continue;
                }
            }
        }
        
        return null;
    }
    
    /**
     * Generate description from SMS
     */
    private String generateDescription(String smsText, String type) {
        String payeeName = extractPayeeOrPayerName(smsText);
        
        StringBuilder description = new StringBuilder();
        
        if (payeeName != null && !payeeName.isEmpty()) {
            if (type != null && type.equals("credit")) {
                description.append("Received from ").append(payeeName);
            } else if (type != null && type.equals("debit")) {
                description.append("Transfer to ").append(payeeName);
            } else {
                description.append("Transaction with ").append(payeeName);
            }
        } else {
            if (type != null && type.equals("credit")) {
                description.append("Money received credit transaction");
            } else if (type != null && type.equals("debit")) {
                description.append("Payment made debit transaction");
            } else {
                description.append("Bank transaction completed");
            }
        }
        
        if (type != null) {
            description.append(" ").append(type).append(" type");
        }
        
        // Ensure 10-15 words
        String[] words = description.toString().split("\\s+");
        if (words.length < 10) {
            description.append(" bank account transaction completed successfully");
        } else if (words.length > 15) {
            StringBuilder trimmed = new StringBuilder();
            for (int i = 0; i < 15; i++) {
                if (i > 0) trimmed.append(" ");
                trimmed.append(words[i]);
            }
            return trimmed.toString();
        }
        
        return description.toString().trim();
    }
    
    /**
     * Extract payee or payer name from SMS
     */
    private String extractPayeeOrPayerName(String smsText) {
        Pattern namePattern1 = Pattern.compile(
            "(?:to|from)\\s+(?:mr|mrs|ms|shri|shrimati)\\.?\\s+([A-Z][A-Z\\s]{2,30})",
            Pattern.CASE_INSENSITIVE
        );
        Matcher matcher1 = namePattern1.matcher(smsText);
        if (matcher1.find()) {
            return matcher1.group(1).trim();
        }
        
        Pattern namePattern2 = Pattern.compile(
            "(?:to|from|payee|payer|beneficiary)[\\s:]+([A-Z][A-Z\\s]{2,30})",
            Pattern.CASE_INSENSITIVE
        );
        Matcher matcher2 = namePattern2.matcher(smsText);
        if (matcher2.find()) {
            return matcher2.group(1).trim();
        }
        
        Pattern upiPattern = Pattern.compile("([a-z0-9._-]+@[a-z]+)", Pattern.CASE_INSENSITIVE);
        Matcher upiMatcher = upiPattern.matcher(smsText);
        if (upiMatcher.find()) {
            return upiMatcher.group(1);
        }
        
        String lowerText = smsText.toLowerCase();
        String[] merchants = {
            "amazon", "flipkart", "swiggy", "zomato", "uber", "ola",
            "paytm", "phonepe", "gpay", "razorpay", "stripe", "netflix"
        };
        
        for (String merchant : merchants) {
            if (lowerText.contains(merchant)) {
                return merchant.substring(0, 1).toUpperCase() + merchant.substring(1);
            }
        }
        
        return null;
    }
    
    /**
     * Warm up the model for faster first inference
     */
    public void warmup() {
        if (ortSession != null && modelReady) {
            executorService.execute(() -> {
                try {
                    // Run a dummy inference to warm up
                    float[] dummyInput = new float[128];
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
        return modelReady;
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
