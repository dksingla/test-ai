package com.example.expensetracker.api;

public class LoginResponse {
    private LoginData data;
    
    public LoginData getData() {
        return data;
    }
    
    public void setData(LoginData data) {
        this.data = data;
    }
    
    public static class LoginData {
        private String message;
        private String email;
        
        public String getMessage() {
            return message;
        }
        
        public void setMessage(String message) {
            this.message = message;
        }
        
        public String getEmail() {
            return email;
        }
        
        public void setEmail(String email) {
            this.email = email;
        }
    }
}

