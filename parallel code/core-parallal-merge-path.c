#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

#define MAX_THREADS 16
#define SEQUENTIAL_THRESHOLD 100000

// Timing function
double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void merge_arrays(int A[], int m, int B[], int n, int S[]) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j])
            S[k++] = A[i++];
        else
            S[k++] = B[j++];
    }
    while (i < m) S[k++] = A[i++];
    while (j < n) S[k++] = B[j++];
}

int find_merge_path_intersection(int A[], int m, int B[], int n, int diagonal_id) {
    int begin = (diagonal_id > m) ? diagonal_id - m : 0;
    int end = (diagonal_id < n) ? diagonal_id : n;
    
    while (begin < end) {
        int mid = begin + (end - begin) / 2;
        int j = diagonal_id - mid;
        
        if (j > 0 && mid < m && B[j-1] > A[mid])
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

void parallel_merge_path(int A[], int m, int B[], int n, int S[]) {
    int num_threads = omp_get_max_threads();
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    if (m + n < SEQUENTIAL_THRESHOLD || num_threads == 1) {
        merge_arrays(A, m, B, n, S);
        return;
    }
    
    int partition_points_A[num_threads + 1];
    int partition_points_B[num_threads + 1];
    int total_size = m + n;
    
    partition_points_A[0] = 0;
    partition_points_B[0] = 0;
    partition_points_A[num_threads] = m;
    partition_points_B[num_threads] = n;
    
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < num_threads; i++) {
        int diagonal = (int)(((long long)i * total_size) / num_threads);
        partition_points_A[i] = find_merge_path_intersection(A, m, B, n, diagonal);
        partition_points_B[i] = diagonal - partition_points_A[i];
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
        int start_a = partition_points_A[i];
        int end_a = partition_points_A[i+1];
        int start_b = partition_points_B[i];
        int end_b = partition_points_B[i+1];
        int start_s = start_a + start_b;
        
        merge_arrays(A + start_a, end_a - start_a, 
                   B + start_b, end_b - start_b, 
                   S + start_s);
    }
}

void generate_sorted_array(int arr[], int size) {
    if (size == 0) return;
    arr[0] = rand() % 10;
    for (int i = 1; i < size; i++)
        arr[i] = arr[i-1] + (rand() % 10 + 1);
}

int main() {
    int m = 1000000, n = 1000000;
    int *A = malloc(m * sizeof(int));
    int *B = malloc(n * sizeof(int));
    int *S = malloc((m+n) * sizeof(int));
    
    if (!A || !B || !S) {
        printf("Memory allocation failed!\n");
        if (A) free(A);
        if (B) free(B);
        if (S) free(S);
        return 1;
    }

    srand(time(NULL));
    
    printf("Generating sorted arrays...\n");
    double gen_start = get_current_time();
    generate_sorted_array(A, m);
    generate_sorted_array(B, n);
    double gen_end = get_current_time();
    printf("Array generation completed in %.4f seconds\n", gen_end - gen_start);

    printf("\nRunning parallel merge with %d threads...\n", omp_get_max_threads());
    double merge_start = get_current_time();
    parallel_merge_path(A, m, B, n, S);
    double merge_end = get_current_time();
    
    printf("\nResults:\n");
    printf("Total elements merged: %d\n", m + n);
    printf("Parallel merge time: %.4f seconds\n", merge_end - merge_start);
    printf("Throughput: %.2f million elements/second\n", 
          (m+n)/((merge_end - merge_start)*1000000));
    
    free(A); free(B); free(S);
    return 0;
}
