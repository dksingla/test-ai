package com.example.expensetracker;

import android.os.Bundle;
import android.util.Log;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.expensetracker.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private ActivityMainBinding binding;
    private GenerativeModelHelper generativeModelHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        BottomNavigationView navView = findViewById(R.id.nav_view);
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_activity_main);
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(binding.navView, navController);

        // Initialize ML Kit GenAI Prompt API
        initializeGenAI();
    }

    /**
     * Initialize the ML Kit GenAI Prompt API
     */
    private void initializeGenAI() {
        generativeModelHelper = new GenerativeModelHelper(this);
        
        // Check model status and prepare if needed
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
                Log.d(TAG, "Model download progress: " + bytesDownloaded + " bytes");
            }

            @Override
            public void onDownloadCompleted() {
                Log.d(TAG, "Model download completed");
            }

            @Override
            public void onDownloadFailed(String error) {
                Log.e(TAG, "Model download failed: " + error);
            }

            @Override
            public void onModelReady() {
                Log.d(TAG, "Model is ready to use");
                // Optional: Warm up the model for faster first inference
                generativeModelHelper.warmup();
                
                // Example: Generate content once model is ready
                exampleTextGeneration();
            }
        });
    }

    /**
     * Example method demonstrating text-only content generation
     */
    private void exampleTextGeneration() {
        String prompt = "Write a 3 sentence story about a magical dog.";
        
        generativeModelHelper.generateContent(prompt, new GenerativeModelHelper.ContentGenerationCallback() {
            @Override
            public void onSuccess(String response) {
                Log.d(TAG, "Generated content: " + response);
                // You can use the response here, e.g., display it in UI
            }

            @Override
            public void onFailure(String error) {
                Log.e(TAG, "Content generation failed: " + error);
            }
        });
    }

    /**
     * Example method demonstrating text generation with optional parameters
     * Note: Optional parameters can be added using GenerateContentRequest.Builder
     * if needed in the future
     */
    private void exampleTextGenerationWithParameters() {
        String prompt = "Write a 3 sentence story about a magical dog.";
        
        // Using basic text generation - optional parameters can be added later
        generativeModelHelper.generateContent(
            prompt,
            new GenerativeModelHelper.ContentGenerationCallback() {
                @Override
                public void onSuccess(String response) {
                    Log.d(TAG, "Generated content with parameters: " + response);
                }

                @Override
                public void onFailure(String error) {
                    Log.e(TAG, "Content generation failed: " + error);
                }
            }
        );
    }

    /**
     * Example method demonstrating multimodal content generation (image + text)
     * Note: You would need to provide a Bitmap image parameter
     */
    /*
    private void exampleMultimodalGeneration(android.graphics.Bitmap image) {
        String textPrompt = "What's in this image?";
        
        generativeModelHelper.generateContent(
            textPrompt,
            image,
            new GenerativeModelHelper.ContentGenerationCallback() {
                @Override
                public void onSuccess(String response) {
                    Log.d(TAG, "Generated multimodal content: " + response);
                }

                @Override
                public void onFailure(String error) {
                    Log.e(TAG, "Multimodal generation failed: " + error);
                }
            }
        );
    }
    */
}