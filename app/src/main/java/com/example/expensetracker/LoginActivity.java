package com.example.expensetracker;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.expensetracker.api.ApiClient;
import com.example.expensetracker.api.LoginRequest;
import com.example.expensetracker.api.LoginResponse;
import com.example.expensetracker.databinding.ActivityLoginBinding;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONObject;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LoginActivity extends AppCompatActivity {

    private static final String TAG = "LoginActivity";
    private ActivityLoginBinding binding;
    private boolean isLoading = false;
    private boolean isAnalyzingSms = false;
    private boolean isModelReady = false;
    private GenerativeModelHelper generativeModelHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            binding = ActivityLoginBinding.inflate(getLayoutInflater());
            setContentView(binding.getRoot());
            
            // Initialize GenerativeModelHelper
            generativeModelHelper = new GenerativeModelHelper(this);
            initializeModel();
            
            setupViews();
        } catch (Exception e) {
            Log.e(TAG, "Error in onCreate", e);
            throw e;
        }
    }
    
    private void initializeModel() {
        try {
            generativeModelHelper.checkAndPrepareModel(new GenerativeModelHelper.ModelStatusCallback() {
                @Override
                public void onStatusChecked(int status) {
                    Log.d(TAG, "Model status checked: " + status);
                }

                @Override
                public void onDownloadStarted() {
                    Log.d(TAG, "Model download started");
                }

                @Override
                public void onDownloadProgress(long bytesDownloaded) {
                    Log.d(TAG, "Model download progress: " + bytesDownloaded);
                }

                @Override
                public void onDownloadCompleted() {
                    Log.d(TAG, "Model download completed");
                }

                @Override
                public void onDownloadFailed(String error) {
                    Log.e(TAG, "Model initialization failed: " + error);
                    isModelReady = false;
                    // Show user-friendly error message
                    Toast.makeText(LoginActivity.this, "AI model failed to load: " + error, Toast.LENGTH_LONG).show();
                    // Disable SMS analysis button if model is not available
                    binding.analyzeSmsButton.setEnabled(false);
                    binding.analyzeSmsButton.setText("AI Not Available");
                }

                @Override
                public void onModelReady() {
                    Log.d(TAG, "Model is ready");
                    isModelReady = true;
                    binding.analyzeSmsButton.setEnabled(true);
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize AI model", e);
            Toast.makeText(this, "Failed to initialize AI: " + e.getMessage(), Toast.LENGTH_LONG).show();
            binding.analyzeSmsButton.setEnabled(false);
            binding.analyzeSmsButton.setText("AI Not Available");
        }
    }

    private void setupViews() {
        // Back button
        binding.backButton.setOnClickListener(v -> {
            finish();
        });

        // Email input watcher
        binding.emailInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updateButtonState();
            }

            @Override
            public void afterTextChanged(Editable s) {}
        });

        // Send OTP button
        binding.sendOtpButton.setOnClickListener(v -> {
            handleSendOTP();
        });

        // Sign up link
        binding.signUpLink.setOnClickListener(v -> {
            Intent intent = new Intent(LoginActivity.this, SignUpActivity.class);
            startActivity(intent);
        });

        // Analyze SMS button
        binding.analyzeSmsButton.setOnClickListener(v -> {
            handleAnalyzeSms();
        });

        // Request focus on email input after view is laid out
        binding.emailInput.post(() -> {
            binding.emailInput.requestFocus();
            // Show keyboard after a short delay
            binding.emailInput.postDelayed(() -> {
                InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.showSoftInput(binding.emailInput, InputMethodManager.SHOW_IMPLICIT);
                }
            }, 100);
        });
    }

    private void updateButtonState() {
        String email = binding.emailInput.getText() != null ? 
                binding.emailInput.getText().toString().trim() : "";
        binding.sendOtpButton.setEnabled(!email.isEmpty() && !isLoading);
    }

    private void handleSendOTP() {
        String email = binding.emailInput.getText() != null ? 
                binding.emailInput.getText().toString().trim() : "";

        if (email.isEmpty()) {
            return;
        }

        isLoading = true;
        updateButtonState();
        binding.sendOtpButton.setText(R.string.sending);
        binding.progressBar.setVisibility(View.VISIBLE);

        // Hide keyboard
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) {
            imm.hideSoftInputFromWindow(binding.emailInput.getWindowToken(), 0);
        }

        LoginRequest request = new LoginRequest(email);
        ApiClient.getApiService().login(request).enqueue(new Callback<LoginResponse>() {
            @Override
            public void onResponse(Call<LoginResponse> call, Response<LoginResponse> response) {
                isLoading = false;
                binding.progressBar.setVisibility(View.GONE);
                binding.sendOtpButton.setText(R.string.send_otp);
                updateButtonState();

                if (response.isSuccessful() && response.body() != null) {
                    LoginResponse loginResponse = response.body();
                    LoginResponse.LoginData data = loginResponse.getData();
                    if (data != null && data.getMessage() != null) {
                        // OTP has been sent, navigate to OTP verification screen
                        Intent intent = new Intent(LoginActivity.this, OtpActivity.class);
                        intent.putExtra("email", email);
                        startActivity(intent);
                    } else {
                        Toast.makeText(LoginActivity.this, "Unexpected response format", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    String errorMsg = "Login failed. Please try again.";
                    if (response.errorBody() != null) {
                        try {
                            errorMsg = response.errorBody().string();
                        } catch (Exception e) {
                            // Use default message
                        }
                    }
                    Toast.makeText(LoginActivity.this, errorMsg, Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<LoginResponse> call, Throwable t) {
                isLoading = false;
                binding.progressBar.setVisibility(View.GONE);
                binding.sendOtpButton.setText(R.string.send_otp);
                updateButtonState();
                
                Toast.makeText(LoginActivity.this, "Error: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }
    
    private void handleAnalyzeSms() {
        String smsText = binding.smsInput.getText() != null ? 
                binding.smsInput.getText().toString().trim() : "";

        if (smsText.isEmpty()) {
            Toast.makeText(this, "Please enter SMS text", Toast.LENGTH_SHORT).show();
            return;
        }

        if (isAnalyzingSms) {
            return;
        }
        
        // ✅ SAFE TO CALL MODEL NOW - Only call after onModelReady() fires
        if (!isModelReady || !generativeModelHelper.isModelReady()) {
            Toast.makeText(this, "AI model is not ready. Please wait.", Toast.LENGTH_LONG).show();
            return;
        }

        isAnalyzingSms = true;
        binding.analyzeSmsButton.setEnabled(false);
        binding.aiProgressBar.setVisibility(View.VISIBLE);
        binding.resultsCard.setVisibility(View.GONE);

        // Hide keyboard
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) {
            View currentFocus = getCurrentFocus();
            if (currentFocus != null) {
                imm.hideSoftInputFromWindow(currentFocus.getWindowToken(), 0);
            }
        }

        // Generate content using ONNX models (expects raw SMS text)
        generativeModelHelper.generateContent(smsText, new GenerativeModelHelper.ContentGenerationCallback() {
            @Override
            public void onSuccess(String response) {
                isAnalyzingSms = false;
                binding.analyzeSmsButton.setEnabled(true);
                binding.aiProgressBar.setVisibility(View.GONE);

                try {
                    // Parse JSON response
                    // Sometimes AI might return text with markdown code blocks, so we need to extract JSON
                    String jsonString = response.trim();
                    if (jsonString.startsWith("```json")) {
                        jsonString = jsonString.substring(7);
                    }
                    if (jsonString.startsWith("```")) {
                        jsonString = jsonString.substring(3);
                    }
                    if (jsonString.endsWith("```")) {
                        jsonString = jsonString.substring(0, jsonString.length() - 3);
                    }
                    jsonString = jsonString.trim();

                    // Find JSON object in the response
                    int jsonStart = jsonString.indexOf("{");
                    int jsonEnd = jsonString.lastIndexOf("}");
                    if (jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart) {
                        jsonString = jsonString.substring(jsonStart, jsonEnd + 1);
                    }

                    JSONObject jsonObject = new JSONObject(jsonString);
                    
                    String amount = jsonObject.optString("amount", "N/A");
                    String type = jsonObject.optString("type", "N/A");
                    String description = jsonObject.optString("description", "N/A");

                    // Display results
                    binding.amountValue.setText(amount);
                    binding.typeValue.setText(type);
                    binding.descriptionValue.setText(description);
                    binding.resultsCard.setVisibility(View.VISIBLE);
                } catch (Exception e) {
                    Log.e(TAG, "Error parsing JSON response", e);
                    Toast.makeText(LoginActivity.this, "Error parsing response: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    Log.d(TAG, "Raw response: " + response);
                }
            }

            @Override
            public void onFailure(String error) {
                isAnalyzingSms = false;
                binding.analyzeSmsButton.setEnabled(true);
                binding.aiProgressBar.setVisibility(View.GONE);
                
                Toast.makeText(LoginActivity.this, "Analysis failed: " + error, Toast.LENGTH_LONG).show();
                Log.e(TAG, "SMS analysis failed: " + error);
            }
        });
    }
}

