package com.example.expensetracker;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

import org.json.JSONObject;

/**
 * Helper class to manage TensorFlow Lite model for SMS analysis.
 * Handles model loading, text preprocessing, and inference to extract
 * transaction type (credit/debit) and amount from SMS messages.
 */
public class GenerativeModelHelper {
    private static final String TAG = "GenerativeModelHelper";
    private static final String MODEL_FILE = "sms_model.tflite";
    
    // Model parameters (from Python training code)
    private static final int MAX_SEQUENCE_LENGTH = 40;
    private static final int VOCAB_SIZE = 800;
    private static final String OOV_TOKEN = "<OOV>";
    private static final int OOV_TOKEN_ID = 1; // Typically OOV is token 1
    
    private Interpreter tfliteInterpreter;
    private Context context;
    private boolean modelReady = false;
    private Tokenizer tokenizer;
    
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
        Log.d(TAG, "🔍 TFLITE_MODEL: GenerativeModelHelper constructor called");
        this.context = context;
        this.tokenizer = new Tokenizer(context);
        initializeModel();
    }
    
    /**
     * Initialize the TensorFlow Lite model
     */
    private void initializeModel() {
        Log.d(TAG, "🔍 TFLITE_MODEL: Initializing TensorFlow Lite model...");
        try {
            ByteBuffer modelBuffer = loadModelFile();
            // Create Interpreter without Options to avoid InterpreterApi dependency issue
            // Default settings will be used (single thread, no delegates)
            tfliteInterpreter = new Interpreter(modelBuffer);
            Log.d(TAG, "🔍 TFLITE_MODEL: ✅ Model loaded successfully");
            modelReady = true;
        } catch (IllegalArgumentException e) {
            Log.e(TAG, "🔍 TFLITE_MODEL: ❌ Model compatibility error", e);
            Log.e(TAG, "🔍 TFLITE_MODEL: This usually means the model was created with a newer TensorFlow version.");
            Log.e(TAG, "🔍 TFLITE_MODEL: Solution: Re-export the model with compatibility settings:");
            Log.e(TAG, "🔍 TFLITE_MODEL: converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]");
            modelReady = false;
        } catch (Exception e) {
            Log.e(TAG, "🔍 TFLITE_MODEL: ❌ Failed to initialize model", e);
            modelReady = false;
        }
    }
    
    /**
     * Load model file from assets
     */
    private ByteBuffer loadModelFile() throws IOException {
        Log.d(TAG, "🔍 TFLITE_MODEL: Loading model from assets: " + MODEL_FILE);
        InputStream inputStream = context.getAssets().open(MODEL_FILE);
        
        // Read all bytes from the input stream
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int len;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        inputStream.close();
        
        byte[] modelBytes = byteBuffer.toByteArray();
        
        // Create ByteBuffer from byte array
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        modelBuffer.order(ByteOrder.nativeOrder());
        modelBuffer.put(modelBytes);
        modelBuffer.rewind();
        
        Log.d(TAG, "🔍 TFLITE_MODEL: Model file loaded, size: " + modelBytes.length + " bytes");
        return modelBuffer;
    }
    
    /**
     * Check and prepare model (for compatibility with existing interface)
     */
    public void checkAndPrepareModel(ModelStatusCallback callback) {
        Log.d(TAG, "🔍 TFLITE_MODEL: checkAndPrepareModel() called");
        
        if (tfliteInterpreter == null) {
            Log.d(TAG, "🔍 TFLITE_MODEL: Model not loaded, attempting to initialize...");
            try {
                initializeModel();
            } catch (Exception e) {
                Log.e(TAG, "🔍 TFLITE_MODEL: ❌ Failed to initialize model", e);
                callback.onDownloadFailed("Failed to load model: " + e.getMessage());
                return;
            }
        }
        
        if (modelReady) {
            Log.i(TAG, "🔍 TFLITE_MODEL: ✅ Model is ready");
            callback.onStatusChecked(1); // FEATURE_STATUS_AVAILABLE
            callback.onModelReady();
        } else {
            Log.e(TAG, "🔍 TFLITE_MODEL: ❌ Model is not ready");
            callback.onDownloadFailed("Model failed to load");
        }
    }
    
    /**
     * Generate content from text-only input (SMS analysis)
     */
    public void generateContent(String prompt, ContentGenerationCallback callback) {
        generateContent(prompt, null, callback);
    }
    
    /**
     * Generate content from multimodal input (image + text)
     * Note: Image input is not supported with TFLite model, only SMS text analysis
     */
    public void generateContent(String textPrompt, Bitmap image, ContentGenerationCallback callback) {
        if (image != null) {
            Log.w(TAG, "🔍 TFLITE_MODEL: Image input not supported, ignoring image");
        }
        
        if (!modelReady || tfliteInterpreter == null) {
            callback.onFailure("Model is not ready. Please wait for model initialization.");
            return;
        }
        
        if (!tokenizer.isInitialized()) {
            callback.onFailure("Tokenizer failed to load. Please check tokenizer.json in assets.");
            return;
        }
        
        Log.d(TAG, "🔍 TFLITE_MODEL: Analyzing SMS text: " + textPrompt);
        
        try {
            // Extract SMS text from prompt if it contains "SMS: " prefix
            String smsText = textPrompt;
            if (textPrompt.contains("SMS: \"")) {
                int start = textPrompt.indexOf("SMS: \"") + 6;
                int end = textPrompt.indexOf("\"", start);
                if (end > start) {
                    smsText = textPrompt.substring(start, end);
                }
            }
            
            // Preprocess SMS text
            int[] inputSequence = preprocessText(smsText);
            Log.d(TAG, "🔍 TFLITE_MODEL: Preprocessed sequence length: " + inputSequence.length);
            Log.d(TAG, "🔍 TFLITE_MODEL: Preprocessed sequence: " + Arrays.toString(inputSequence));
            
            // Prepare input/output buffers
            // Model expects INT32 input for Embedding layer
            int[][] inputBuffer = new int[1][MAX_SEQUENCE_LENGTH];
            float[][] typeOutput = new float[1][1];
            float[][] amountOutput = new float[1][1];
            
            // Copy sequence to input buffer (INT32)
            for (int i = 0; i < Math.min(inputSequence.length, MAX_SEQUENCE_LENGTH); i++) {
                inputBuffer[0][i] = inputSequence[i];
            }
            
            // Run inference
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, typeOutput);
            outputs.put(1, amountOutput);
            
            Log.d(TAG, "🔍 TFLITE_MODEL: Running inference...");
            tfliteInterpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
            
            // Parse results
            float typeProbability = typeOutput[0][0];
            float amount = amountOutput[0][0];
            
            Log.d(TAG, "🔍 TFLITE_MODEL: Inference results - Type probability: " + typeProbability + ", Amount: " + amount);
            
            // Convert to readable format
            String type = typeProbability >= 0.5 ? "debit" : "credit";
            DecimalFormat df = new DecimalFormat("#.##");
            String amountStr = "₹" + df.format(Math.abs(amount));
            
            // Generate description
            String description = generateDescription(smsText, type, amount);
            
            // Create JSON response
            JSONObject jsonResponse = new JSONObject();
            jsonResponse.put("type", type);
            jsonResponse.put("amount", amountStr);
            jsonResponse.put("description", description);
            
            String jsonString = jsonResponse.toString();
            Log.d(TAG, "🔍 TFLITE_MODEL: ✅ Analysis complete: " + jsonString);
            callback.onSuccess(jsonString);
            
        } catch (Exception e) {
            Log.e(TAG, "🔍 TFLITE_MODEL: ❌ Analysis failed", e);
            callback.onFailure("Analysis failed: " + e.getMessage());
        }
    }
    
    /**
     * Preprocess text: tokenize and pad to MAX_SEQUENCE_LENGTH
     */
    private int[] preprocessText(String text) {
        // Tokenize text
        List<Integer> tokens = tokenizer.textsToSequences(text);
        
        // Pad or truncate to MAX_SEQUENCE_LENGTH
        int[] sequence = new int[MAX_SEQUENCE_LENGTH];
        Arrays.fill(sequence, 0); // Padding with 0
        
        int length = Math.min(tokens.size(), MAX_SEQUENCE_LENGTH);
        for (int i = 0; i < length; i++) {
            sequence[i] = tokens.get(i);
        }
        
        return sequence;
    }
    
    /**
     * Generate description from SMS text, type, and amount
     */
    private String generateDescription(String smsText, String type, float amount) {
        // Extract key information from SMS
        String lowerSms = smsText.toLowerCase();
        
        // Try to find merchant/payee name
        String merchant = extractMerchantName(smsText);
        
        // Build description
        String action = type.equals("credit") ? "received from" : "paid to";
        DecimalFormat df = new DecimalFormat("#.##");
        String amountStr = "₹" + df.format(Math.abs(amount));
        
        if (!merchant.isEmpty()) {
            return String.format(Locale.getDefault(), "%s %s %s transaction", 
                merchant, action, amountStr);
        } else {
            return String.format(Locale.getDefault(), "Bank %s %s transaction", 
                type, amountStr);
        }
    }
    
    /**
     * Extract merchant/payee name from SMS (simple heuristic)
     */
    private String extractMerchantName(String smsText) {
        // Common patterns in Indian SMS
        String[] patterns = {
            "to\\s+([A-Z][A-Za-z\\s]+?)\\s+(?:UPI|bank|account)",
            "from\\s+([A-Z][A-Za-z\\s]+?)\\s+(?:UPI|bank|account)",
            "paid\\s+to\\s+([A-Z][A-Za-z\\s]+?)",
            "received\\s+from\\s+([A-Z][A-Za-z\\s]+?)",
        };
        
        for (String pattern : patterns) {
            Pattern p = Pattern.compile(pattern, Pattern.CASE_INSENSITIVE);
            java.util.regex.Matcher m = p.matcher(smsText);
            if (m.find()) {
                String name = m.group(1).trim();
                if (name.length() > 2 && name.length() < 30) {
                    return name;
                }
            }
        }
        
        return "";
    }
    
    /**
     * Check if model is ready to use
     */
    public boolean isModelReady() {
        Log.d(TAG, "🔍 TFLITE_MODEL: isModelReady() called - returning: " + modelReady);
        return modelReady;
    }
    
    /**
     * Check AICore availability (for compatibility - not needed for TFLite)
     */
    public void checkAICoreAvailability(ModelStatusCallback callback) {
        Log.d(TAG, "🔍 TFLITE_MODEL: checkAICoreAvailability() called (not needed for TFLite)");
        checkAndPrepareModel(callback);
    }
    
    /**
     * Warm up the model for faster first inference
     */
    public void warmup() {
        if (tfliteInterpreter != null && modelReady) {
            Log.d(TAG, "🔍 TFLITE_MODEL: Warming up model...");
            try {
                // Run a dummy inference to warm up
                // Use INT32 to match model input type (Embedding layer)
                int[][] dummyInput = new int[1][MAX_SEQUENCE_LENGTH];
                float[][] dummyTypeOutput = new float[1][1];
                float[][] dummyAmountOutput = new float[1][1];
                
                Map<Integer, Object> outputs = new HashMap<>();
                outputs.put(0, dummyTypeOutput);
                outputs.put(1, dummyAmountOutput);
                
                tfliteInterpreter.runForMultipleInputsOutputs(new Object[]{dummyInput}, outputs);
                Log.d(TAG, "🔍 TFLITE_MODEL: ✅ Model warmed up");
            } catch (Exception e) {
                Log.e(TAG, "🔍 TFLITE_MODEL: Failed to warm up model", e);
            }
        }
    }
    
    /**
     * Tokenizer implementation that loads word_index from tokenizer.json
     * Uses the exact vocabulary from Python Keras tokenizer training
     */
    private class Tokenizer {
        private static final String TAG = "Tokenizer";
        private static final String TOKENIZER_FILE = "tokenizer.json";
        private static final Pattern WORD_PATTERN = Pattern.compile("\\b\\w+\\b");
        private Map<String, Integer> wordToIndex = new HashMap<>();
        private final int OOV_TOKEN_ID = 1; // <OOV> token ID from tokenizer.json
        private boolean initialized = false;
        private Context context;
        
        public Tokenizer(Context context) {
            this.context = context;
            loadTokenizer();
        }
        
        /**
         * Load tokenizer.json from assets and parse word_index
         */
        private void loadTokenizer() {
            try {
                Log.d(TAG, "Loading tokenizer from assets: " + TOKENIZER_FILE);
                InputStream inputStream = context.getAssets().open(TOKENIZER_FILE);
                
                // Read entire file
                ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
                byte[] buffer = new byte[1024];
                int len;
                while ((len = inputStream.read(buffer)) != -1) {
                    byteBuffer.write(buffer, 0, len);
                }
                inputStream.close();
                
                String jsonString = byteBuffer.toString("UTF-8");
                Log.d(TAG, "Tokenizer JSON loaded, size: " + jsonString.length() + " bytes");
                
                // Parse JSON
                JSONObject tokenizerJson = new JSONObject(jsonString);
                JSONObject config = tokenizerJson.getJSONObject("config");
                
                // word_index is stored as a JSON string inside config, so we need to parse it again
                String wordIndexJsonString = config.getString("word_index");
                JSONObject wordIndexJson = new JSONObject(wordIndexJsonString);
                
                // Build wordToIndex map
                wordToIndex.clear();
                java.util.Iterator<String> keys = wordIndexJson.keys();
                while (keys.hasNext()) {
                    String word = keys.next();
                    int index = wordIndexJson.getInt(word);
                    wordToIndex.put(word.toLowerCase(), index);
                }
                
                initialized = true;
                Log.d(TAG, "✅ Tokenizer loaded successfully. Vocabulary size: " + wordToIndex.size());
                Log.d(TAG, "OOV token ID: " + OOV_TOKEN_ID);
                
            } catch (IOException e) {
                Log.e(TAG, "❌ Failed to load tokenizer.json from assets", e);
                initialized = false;
            } catch (Exception e) {
                Log.e(TAG, "❌ Failed to parse tokenizer.json", e);
                initialized = false;
            }
        }
        
        /**
         * Convert text to sequence of token IDs
         * Matches Keras Tokenizer behavior: lowercase, split by space, filter punctuation
         */
        public List<Integer> textsToSequences(String text) {
            if (!initialized) {
                Log.w(TAG, "Tokenizer not initialized, using OOV token");
                return Arrays.asList(OOV_TOKEN_ID);
            }
            
            List<Integer> sequences = new ArrayList<>();
            
            // Preprocess text: lowercase and split by space (matching Keras tokenizer config)
            String lowerText = text.toLowerCase();
            
            // Extract words using regex (matches word boundaries)
            java.util.regex.Matcher matcher = WORD_PATTERN.matcher(lowerText);
            while (matcher.find()) {
                String word = matcher.group();
                Integer tokenId = wordToIndex.get(word);
                
                if (tokenId != null) {
                    sequences.add(tokenId);
                } else {
                    // Word not in vocabulary - use OOV token
                    sequences.add(OOV_TOKEN_ID);
                    Log.v(TAG, "OOV word: " + word);
                }
            }
            
            return sequences;
        }
        
        /**
         * Check if tokenizer is initialized
         */
        public boolean isInitialized() {
            return initialized;
        }
    }
    
    /**
     * Cleanup resources
     */
    public void close() {
        if (tfliteInterpreter != null) {
            tfliteInterpreter.close();
            tfliteInterpreter = null;
            modelReady = false;
            Log.d(TAG, "🔍 TFLITE_MODEL: Model interpreter closed");
        }
    }
}

