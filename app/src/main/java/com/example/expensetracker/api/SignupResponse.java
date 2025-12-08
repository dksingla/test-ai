package com.example.expensetracker.api;

public class SignupResponse {
    private SignupData data;
    
    public SignupData getData() {
        return data;
    }
    
    public void setData(SignupData data) {
        this.data = data;
    }
    
    public static class SignupData {
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

