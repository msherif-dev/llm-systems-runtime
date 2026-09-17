// Reduction in CPU 

// Reduction For Sum 

float reduce_sum_cpu(float*x , int n){

    float sum = 0.0f;

    for(int i = 0 ; i  < n ; i++){
        sum+=x[i];
    }

    return sum ; 
}

// Reduction For Max 

float reduce_max_cpu(float*x , int n ){
    float max_value = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }
    return max_value;
}