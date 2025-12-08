package com.example.expensetracker;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.expensetracker.api.ApiClient;
import com.example.expensetracker.api.SignupRequest;
import com.example.expensetracker.api.SignupResponse;
import com.example.expensetracker.databinding.ActivitySignupBinding;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class SignUpActivity extends AppCompatActivity {

    private ActivitySignupBinding binding;
    private boolean isLoading = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            binding = ActivitySignupBinding.inflate(getLayoutInflater());
            setContentView(binding.getRoot());
            setupViews();
        } catch (Exception e) {
            android.util.Log.e("SignUpActivity", "Error in onCreate", e);
            throw e;
        }
    }

    private void setupViews() {
        // Back button
        binding.backButton.setOnClickListener(v -> {
            finish();
        });

        // Name input watcher
        binding.nameInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updateButtonState();
            }

            @Override
            public void afterTextChanged(Editable s) {}
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

        // Sign up button
        binding.signUpButton.setOnClickListener(v -> {
            handleSignUp();
        });

        // Login link
        binding.loginLink.setOnClickListener(v -> {
            finish();
        });

        // Request focus on name input after view is laid out
        binding.nameInput.post(() -> {
            binding.nameInput.requestFocus();
            // Show keyboard after a short delay
            binding.nameInput.postDelayed(() -> {
                InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.showSoftInput(binding.nameInput, InputMethodManager.SHOW_IMPLICIT);
                }
            }, 100);
        });
    }

    private void updateButtonState() {
        String name = binding.nameInput.getText() != null ? 
                binding.nameInput.getText().toString().trim() : "";
        String email = binding.emailInput.getText() != null ? 
                binding.emailInput.getText().toString().trim() : "";
        binding.signUpButton.setEnabled(!name.isEmpty() && !email.isEmpty() && !isLoading);
    }

    private void handleSignUp() {
        String name = binding.nameInput.getText() != null ? 
                binding.nameInput.getText().toString().trim() : "";
        String email = binding.emailInput.getText() != null ? 
                binding.emailInput.getText().toString().trim() : "";

        if (name.isEmpty() || email.isEmpty()) {
            return;
        }

        isLoading = true;
        updateButtonState();
        binding.signUpButton.setText(R.string.signing_up);
        binding.progressBar.setVisibility(View.VISIBLE);

        // Hide keyboard
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) {
            View currentFocus = getCurrentFocus();
            if (currentFocus != null) {
                imm.hideSoftInputFromWindow(currentFocus.getWindowToken(), 0);
            }
        }

        SignupRequest request = new SignupRequest(email, name);
        ApiClient.getApiService().signup(request).enqueue(new Callback<SignupResponse>() {
            @Override
            public void onResponse(Call<SignupResponse> call, Response<SignupResponse> response) {
                isLoading = false;
                binding.progressBar.setVisibility(View.GONE);
                binding.signUpButton.setText(R.string.sign_up);
                updateButtonState();

                if (response.isSuccessful() && response.body() != null) {
                    SignupResponse signupResponse = response.body();
                    SignupResponse.SignupData data = signupResponse.getData();
                    if (data != null && data.getMessage() != null) {
                        // OTP has been sent, navigate to OTP verification screen
                        Intent intent = new Intent(SignUpActivity.this, OtpActivity.class);
                        intent.putExtra("email", email);
                        startActivity(intent);
                    } else {
                        Toast.makeText(SignUpActivity.this, "Unexpected response format", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    String errorMsg = "Sign up failed. Please try again.";
                    if (response.errorBody() != null) {
                        try {
                            errorMsg = response.errorBody().string();
                        } catch (Exception e) {
                            // Use default message
                        }
                    }
                    Toast.makeText(SignUpActivity.this, errorMsg, Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<SignupResponse> call, Throwable t) {
                isLoading = false;
                binding.progressBar.setVisibility(View.GONE);
                binding.signUpButton.setText(R.string.sign_up);
                updateButtonState();
                
                Toast.makeText(SignUpActivity.this, "Error: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }
}

