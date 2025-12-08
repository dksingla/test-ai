package com.example.expensetracker.api;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ApiService {
    
    @POST("auth/login")
    Call<LoginResponse> login(@Body LoginRequest request);
    
    @POST("auth/signup")
    Call<SignupResponse> signup(@Body SignupRequest request);
}

