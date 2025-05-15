#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>  // For gettimeofday()

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

void generate_sorted_array(int arr[], int size) {
    if (size == 0) return;
    arr[0] = rand() % 10;
    for (int i = 1; i < size; i++)
        arr[i] = arr[i-1] + (rand() % 10 + 1);
}

double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
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
    printf("Generating arrays...\n");
    generate_sorted_array(A, m);
    generate_sorted_array(B, n);

    printf("Merging %d + %d elements...\n", m, n);
    double start = get_current_time();
    merge_arrays(A, m, B, n, S);
    double end = get_current_time();
    
    printf("Merge completed in %.4f seconds\n", end - start);
    
    free(A); free(B); free(S);
    return 0;
}
