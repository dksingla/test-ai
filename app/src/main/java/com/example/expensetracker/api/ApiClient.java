package com.example.expensetracker.api;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class ApiClient {
    // Option 1: Use localhost with adb port forwarding (recommended)
    // Run: adb reverse tcp:3000 tcp:3000
    private static final String BASE_URL = "http://localhost:3000/";
    
    // Option 2: Use 10.0.2.2 (if port forwarding doesn't work)
    // private static final String BASE_URL = "http://10.0.2.2:3000/";
    
    // Option 3: Use your computer's IP for physical device
    // private static final String BASE_URL = "http://192.168.1.13:3000/";
    
    // Option 4: Use ngrok URL
    // private static final String BASE_URL = "https://xxxxx.ngrok-free.app/";
    private static ApiService apiService;
    
    public static ApiService getApiService() {
        if (apiService == null) {
            try {
                HttpLoggingInterceptor loggingInterceptor = new HttpLoggingInterceptor();
                loggingInterceptor.setLevel(HttpLoggingInterceptor.Level.BODY);
                
                OkHttpClient client = new OkHttpClient.Builder()
                        .addInterceptor(loggingInterceptor)
                        .build();
                
                Retrofit retrofit = new Retrofit.Builder()
                        .baseUrl(BASE_URL)
                        .client(client)
                        .addConverterFactory(GsonConverterFactory.create())
                        .build();
                
                apiService = retrofit.create(ApiService.class);
            } catch (Exception e) {
                android.util.Log.e("ApiClient", "Error creating API service", e);
                throw new RuntimeException("Failed to initialize API client. Please check BASE_URL in ApiClient.java", e);
            }
        }
        return apiService;
    }
}

