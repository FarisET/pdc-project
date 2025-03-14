#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to perform binary search to find the correct partition index
int binary_search(int arr[], int size, int key) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == key)
            return mid;
        else if (arr[mid] < key)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;  // Returns the correct insertion position
}

// Merge two sorted subarrays into output array S
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

// Main function to merge two sorted arrays A and B
void parallel_merge(int A[], int m, int B[], int n, int S[]) {
    // Find median partition index
    int i = m / 2;  // Pick middle of A
    int j = binary_search(B, n, A[i]);  // Find its corresponding index in B

    // Merge two halves in parallel-like fashion
    merge_arrays(A, i, B, j, S); // Left part
    merge_arrays(A + i, m - i, B + j, n - j, S + i + j); // Right part
}

// Function to generate sorted random array
void generate_sorted_array(int arr[], int size) {
    arr[0] = rand() % 10;
    for (int i = 1; i < size; i++)
        arr[i] = arr[i - 1] + (rand() % 10);
}

// Main driver
int main() {
    int m = 2000000000, n = 2000000000; // Size of input arrays
    int *A = (int *)malloc(m * sizeof(int));
    int *B = (int *)malloc(n * sizeof(int));
    int *S = (int *)malloc((m + n) * sizeof(int));

    // Generate sorted arrays
    srand(time(NULL));
    generate_sorted_array(A, m);
    generate_sorted_array(B, n);

    // Measure runtime
    clock_t start = clock();
    parallel_merge(A, m, B, n, S);
    clock_t end = clock();

    // Print runtime
    printf("Merge completed in %f seconds.\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(A);
    free(B);
    free(S);

    return 0;
}
