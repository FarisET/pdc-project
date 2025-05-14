/**
 * Enhanced Parallel Merge Path Implementation
 * 
 * This program implements and benchmarks different merging strategies for sorted arrays:
 * 1. Sequential merge (baseline)
 * 2. Parallel merge using Merge Path algorithm
 * 3. Simple two-way parallel merge
 * 
 * Features:
 * - Support for large array sizes (up to available memory)
 * - Testing with various array types (sorted, reverse sorted, random)
 * - Detailed performance measurements (time, CPU usage, memory, cache)
 * 
 * Based on research by Green et al. on GPU Merge Path algorithm, adapted for OpenMP.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>  // For sysconf
#include <sys/time.h>
#include <sys/resource.h>  // For getrusage
#include <limits.h>

// Define maximum number of threads to test
#define MAX_THREADS 16
// Threshold below which sequential merge is used
#define SEQUENTIAL_THRESHOLD 1000000

// Array type definitions
typedef enum {
    SORTED,
    REVERSE_SORTED,
    RANDOM
} ArrayType;

/**
 * Find the intersection point on the merge path diagonal
 * 
 * This is the core of the Merge Path algorithm. It finds the point where
 * the merge path crosses the diagonal, which determines optimal partition points.
 * 
 * @param A First sorted array
 * @param m Size of first array
 * @param B Second sorted array
 * @param n Size of second array
 * @param diagonal_id The diagonal index to find intersection for
 * @return Index in array A corresponding to the merge path intersection
 */
int find_merge_path_intersection(int A[], int m, int B[], int n, int diagonal_id) {
    // Binary search along the diagonal to find the intersection point
    int begin = (diagonal_id > m) ? diagonal_id - m : 0;
    int end = (diagonal_id < n) ? diagonal_id : n;
    
    while (begin < end) {
        int mid = begin + (end - begin) / 2;
        int j = diagonal_id - mid;
        
        // Check if we found the intersection point
        // If B[j-1] > A[mid], we need to move right on A (increase mid)
        if (j > 0 && mid < m && B[j-1] > A[mid]) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    
    return begin;
}

/**
 * Binary search to find the insertion point of a key in a sorted array
 * 
 * @param arr Sorted array to search in
 * @param size Size of the array
 * @param key Value to find insertion point for
 * @return Index where key should be inserted
 */
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

/**
 * Sequential merge function - baseline implementation
 * 
 * Standard merge of two sorted arrays into a single output array
 * 
 * @param A First sorted array
 * @param m Size of first array
 * @param B Second sorted array
 * @param n Size of second array
 * @param S Output array (size m+n)
 */
void merge_arrays(int A[], int m, int B[], int n, int S[]) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j])
            S[k++] = A[i++];
        else
            S[k++] = B[j++];
    }
    // Copy remaining elements
    while (i < m) S[k++] = A[i++];
    while (j < n) S[k++] = B[j++];
}

/**
 * Parallel merge using the Merge Path algorithm
 * 
 * Divides the merge task into balanced partitions that can be processed
 * independently by multiple threads.
 * 
 * @param A First sorted array
 * @param m Size of first array
 * @param B Second sorted array
 * @param n Size of second array
 * @param S Output array (size m+n)
 */
void parallel_merge_path(int A[], int m, int B[], int n, int S[]) {
    int num_threads = omp_get_max_threads();
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    // For small inputs or single thread, use sequential merge
    if (m + n < SEQUENTIAL_THRESHOLD || num_threads == 1) {
        merge_arrays(A, m, B, n, S);
        return;
    }
    
    // Arrays to store partition points for arrays A and B
    int partition_points_A[num_threads + 1];
    int partition_points_B[num_threads + 1];
    int total_size = m + n;
    
    // First and last partition points are fixed
    partition_points_A[0] = 0;
    partition_points_B[0] = 0;
    partition_points_A[num_threads] = m;
    partition_points_B[num_threads] = n;
    
    // Calculate diagonal intersections for partitioning
    // This determines where each thread will start and end its merge
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < num_threads; i++) {
        // Calculate the diagonal index that represents an equal division of work
        int diagonal = (int)(((long long)i * total_size) / num_threads);
        // Find where this diagonal intersects the merge path
        partition_points_A[i] = find_merge_path_intersection(A, m, B, n, diagonal);
        // Calculate corresponding position in array B
        partition_points_B[i] = diagonal - partition_points_A[i];
    }
    
    // Merge partitioned arrays in parallel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
        // Extract this thread's portion of arrays A and B
        int start_a = partition_points_A[i];
        int end_a = partition_points_A[i+1];
        int start_b = partition_points_B[i];
        int end_b = partition_points_B[i+1];
        int start_s = start_a + start_b;
        
        // Merge this partition
        merge_arrays(A + start_a, end_a - start_a, 
                    B + start_b, end_b - start_b, 
                    S + start_s);
    }
}

/**
 * Simple two-way parallel merge
 * 
 * Splits the work into two halves at a median point and merges in parallel.
 * Simpler than the full Merge Path algorithm.
 * 
 * @param A First sorted array
 * @param m Size of first array
 * @param B Second sorted array
 * @param n Size of second array
 * @param S Output array (size m+n)
 */
void parallel_merge_simple(int A[], int m, int B[], int n, int S[]) {
    // Find median partition index
    int i = m / 2;  // Pick middle of A
    int j = binary_search(B, n, A[i]);  // Find its corresponding index in B
    
    // Merge two halves in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        merge_arrays(A, i, B, j, S); // Left part
        
        #pragma omp section
        merge_arrays(A + i, m - i, B + j, n - j, S + i + j); // Right part
    }
}

/**
 * Generate a sorted array with random increments
 * 
 * @param arr Array to fill with sorted values
 * @param size Size of the array
 */
void generate_sorted_array(int arr[], long long size) {
    if (size == 0) return;

    arr[0] = rand() % 10;  // Start with a small value (0-9)
    if (size == 1) return;

    // Calculate the maximum safe increment per step
    int max_possible_increment = (INT_MAX - arr[0]) / (size - 1);
    if (max_possible_increment <= 0) max_possible_increment = 1;  // Ensure at least +1 per step

    for (long long i = 1; i < size; i++) {
        // Generate a random step (1 to max_possible_increment)
        int step = 1 + (rand() % max_possible_increment);
        arr[i] = arr[i - 1] + step;

        // Safety check (shouldn't trigger if math is correct)
        if (arr[i] < arr[i - 1]) {
            arr[i] = INT_MAX;  // Cap at INT_MAX if overflow occurs
        }
    }
}

/**
 * Generate a reverse sorted array with random decrements
 * 
 * @param arr Array to fill with reverse sorted values
 * @param size Size of the array
 */
void generate_reverse_sorted_array(int arr[], long long size) {
    arr[0] = rand() % 10 + size * 10; // Start with a large number
    for (long long i = 1; i < size; i++)
        arr[i] = arr[i - 1] - (rand() % 10 + 1); // Ensure decreasing
}

/**
 * Comparison function for qsort
 */
int compare_ints(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

/**
 * Generate a random unsorted array
 * 
 * @param arr Array to fill with random values
 * @param size Size of the array
 */
void generate_random_array(int arr[], long long size) {
    for (long long i = 0; i < size; i++)
        arr[i] = rand();
    
    // Sort the array (needed for merge operations)
    qsort(arr, size, sizeof(int), compare_ints);
}

/**
 * Generate arrays based on the requested type
 * 
 * @param A First array to fill
 * @param m Size of first array
 * @param B Second array to fill
 * @param n Size of second array
 * @param type Type of arrays to generate
 */
void generate_arrays(int A[], long long m, int B[], long long n, ArrayType type) {
    switch (type) {
        case SORTED:
            generate_sorted_array(A, m);
            generate_sorted_array(B, n);
            break;
        case REVERSE_SORTED:
            generate_reverse_sorted_array(A, m);
            generate_reverse_sorted_array(B, n);
            // For merge to work, we need to sort these in ascending order
            for (long long i = 0; i < m/2; i++) {
                int temp = A[i];
                A[i] = A[m-i-1];
                A[m-i-1] = temp;
            }
            for (long long i = 0; i < n/2; i++) {
                int temp = B[i];
                B[i] = B[n-i-1];
                B[n-i-1] = temp;
            }
            break;
        case RANDOM:
            generate_random_array(A, m);
            generate_random_array(B, n);
            break;
    }
}

/**
 * Verify that an array is correctly sorted
 * 
 * @param arr Array to check
 * @param size Size of the array
 * @return 1 if sorted, 0 if not sorted
 */
int verify_sort(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
            printf("Sort verification failed at index %d: %d > %d\n", i, arr[i-1], arr[i]);
            return 0;
        }
    }
    return 1;
}

/**
 * Get memory usage in MB
 * 
 * @return Memory usage in MB
 */
double get_memory_usage() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    // Return in MB
    return r_usage.ru_maxrss / 1024.0;
}

/**
 * Get CPU usage
 * 
 * @return CPU usage percentage
 */
double get_cpu_usage() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    double user_time = r_usage.ru_utime.tv_sec + r_usage.ru_utime.tv_usec / 1000000.0;
    double sys_time = r_usage.ru_stime.tv_sec + r_usage.ru_stime.tv_usec / 1000000.0;
    
    return user_time + sys_time;
}

/**
 * Get cache performance metrics (approximated through page faults)
 * 
 * @return Page faults count
 */
long get_page_faults() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return r_usage.ru_majflt;
}

/**
 * Run benchmarks on all merge implementations
 * 
 * Tests sequential merge, parallel Merge Path, and simple two-way parallel merge
 * with various thread counts.
 * 
 * @param A First sorted array
 * @param m Size of first array
 * @param B Second sorted array
 * @param n Size of second array
 * @param array_type_name Name of the array type being tested
 */
void run_benchmarks(int A[], long long m, int B[], long long n, const char* array_type_name) {
    printf("\n===== ARRAY TYPE: %s =====\n", array_type_name);
    printf("Array sizes: A=%lld, B=%lld, Total=%lld\n", m, n, m+n);
    
    // Allocate output arrays for different algorithms
    int *S_seq = (int *)malloc((m + n) * sizeof(int));
    int *S_par = (int *)malloc((m + n) * sizeof(int));
    
    if (!S_seq || !S_par) {
        printf("Memory allocation for output arrays failed\n");
        if (S_seq) free(S_seq);
        if (S_par) free(S_par);
        return;
    }
    
    // Performance metrics before algorithm execution
    double initial_memory = get_memory_usage();
    double initial_cpu = get_cpu_usage();
    long initial_page_faults = get_page_faults();
    
    // Test sequential merge first as baseline
    double start = omp_get_wtime();
    merge_arrays(A, m, B, n, S_seq);
    double end = omp_get_wtime();
    double seq_time = end - start;
    
    // Collect performance metrics
    double seq_memory = get_memory_usage() - initial_memory;
    double seq_cpu = get_cpu_usage() - initial_cpu;
    long seq_page_faults = get_page_faults() - initial_page_faults;
    
    printf("Sequential merge: %.4f seconds\n", seq_time);
    
    // Verify sequential merge is correct
    if (!verify_sort(S_seq, m+n)) {
        printf("Sequential merge failed verification!\n");
        free(S_seq);
        free(S_par);
        return;
    }
    
    // Test with different thread counts
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    int max_threads = omp_get_max_threads();
    
    printf("\n--- Merge Path Implementation ---\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s | %-10s | %s\n", 
           "Threads", "Time (s)", "Speedup", "Efficiency", "Memory (MB)", "CPU usage", "Page faults");
    printf("-----------------------------------------------------------------------------------------\n");
    
    printf("%-8s | %-10.4f | %-10s | %-10s | %-10.2f | %-10.2f | %ld\n", 
           "Seq", seq_time, "-", "-", seq_memory, seq_cpu, seq_page_faults);
    
    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        if (threads > max_threads) {
            printf("%-8d | %-10s | %-10s | %-10s | %-10s | %-10s | %s\n", 
                   threads, "SKIPPED", "-", "-", "-", "-", "Exceeds system threads");
            continue;
        }
        
        omp_set_num_threads(threads);
        
        // Reset performance counters
        initial_memory = get_memory_usage();
        initial_cpu = get_cpu_usage();
        initial_page_faults = get_page_faults();
        
        // Warm up run (not timed)
        parallel_merge_path(A, m, B, n, S_par);
        
        // Actual timed run
        start = omp_get_wtime();
        parallel_merge_path(A, m, B, n, S_par);
        end = omp_get_wtime();
        double par_time = end - start;
        
        // Collect performance metrics
        double par_memory = get_memory_usage() - initial_memory;
        double par_cpu = get_cpu_usage() - initial_cpu;
        long par_page_faults = get_page_faults() - initial_page_faults;
        
        // Verify results against sequential merge
        int matches = 1;
        for (int i = 0; i < m+n; i++) {
            if (S_seq[i] != S_par[i]) {
                matches = 0;
                printf("Mismatch at index %d: Expected %d, Got %d\n", i, S_seq[i], S_par[i]);
                break;
            }
        }
        
        // Calculate speedup and efficiency correctly
        double speedup = seq_time / par_time;
        double efficiency = (speedup / threads) * 100;
        
        printf("%-8d | %-10.4f | %-10.2f | %-10.2f%% | %-10.2f | %-10.2f | %ld %s\n", 
               threads, par_time, speedup, efficiency, par_memory, par_cpu, par_page_faults,
               matches ? "" : "FAILED");
    }
    
    // Test simple two-way parallel merge
    printf("\n--- Simple Two-Way Parallel Merge ---\n");
    printf("%-8s | %-10s | %-10s | %-10s | %-10s | %-10s | %s\n", 
           "Threads", "Time (s)", "Speedup", "Efficiency", "Memory (MB)", "CPU usage", "Page faults");
    printf("-----------------------------------------------------------------------------------------\n");
    
    // Use maximum available threads for simple parallel merge
    omp_set_num_threads(max_threads > 2 ? 2 : max_threads); // This only needs 2 threads
    
    // Reset performance counters
    initial_memory = get_memory_usage();
    initial_cpu = get_cpu_usage();
    initial_page_faults = get_page_faults();
    
    start = omp_get_wtime();
    parallel_merge_simple(A, m, B, n, S_par);
    end = omp_get_wtime();
    double simple_time = end - start;
    
    // Collect performance metrics
    double simple_memory = get_memory_usage() - initial_memory;
    double simple_cpu = get_cpu_usage() - initial_cpu;
    long simple_page_faults = get_page_faults() - initial_page_faults;
    
    // Verify results
    int matches = 1;
    for (int i = 0; i < m+n; i++) {
        if (S_seq[i] != S_par[i]) {
            matches = 0;
            break;
        }
    }
    
    // Calculate speedup and efficiency correctly
    double speedup = seq_time / simple_time;
    double efficiency = (speedup / 2) * 100; // Assuming 2 threads for simple merge
    
    printf("%-8s | %-10.4f | %-10.2f | %-10.2f%% | %-10.2f | %-10.2f | %ld %s\n", 
           "2-way", simple_time, speedup, efficiency, simple_memory, simple_cpu, simple_page_faults,
           matches ? "" : "FAILED");
    
    free(S_seq);
    free(S_par);
}

/**
 * Calculate the maximum array size based on available memory
 * 
 * @return Maximum size of each array that can be safely allocated
 */
long long calculate_max_array_size() {
    // Get physical memory info
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long long total_physical_memory = (long long)pages * page_size;
    
    // Reserve only 75% of available physical memory for our arrays
    // This allows for OS and other program overhead
    long long usable_memory = (total_physical_memory * 3) / 4;
    
    // We need memory for two input arrays and one output array
    // Each element is an int (4 bytes)
    long long max_total_elements = usable_memory / (3 * sizeof(int));
    
    // Since we have two input arrays of equal size, each can hold half
    long long max_per_array = max_total_elements / 2;
    
    return max_per_array;
}

/**
 * Print memory information
 */
void print_memory_info() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long long total_physical_memory = (long long)pages * page_size;
    
    printf("System information:\n");
    printf("Total physical memory: %.2f GB\n", total_physical_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Max available threads: %d\n", omp_get_max_threads());
    
    long long max_per_array = calculate_max_array_size();
    printf("Maximum recommended array size: %lld elements (%.2f GB)\n", 
           max_per_array, 
           (max_per_array * sizeof(int)) / (1024.0 * 1024.0 * 1024.0));
}

/**
 * Main function
 * 
 * Initializes arrays, runs benchmarks, and reports results
 */
int main(int argc, char *argv[]) {
    // Print memory information
    print_memory_info();
    
    // Calculate the maximum possible array size
    long long max_array_size = calculate_max_array_size();
    
    // Get array sizes from command line or use defaults
    long long m = 1000000000, n = 1000000000; // Default sizes
    int array_type = 0; // 0=sorted, 1=reverse sorted, 2=random
    
    if (argc >= 3) {
        m = atoll(argv[1]);
        n = atoll(argv[2]);
        
        if (argc >= 4) {
            array_type = atoi(argv[3]);
            if (array_type < 0 || array_type > 2) {
                printf("Invalid array type. Using default (sorted).\n");
                array_type = 0;
            }
        }
    } else {
        printf("Usage: %s <size_A> <size_B> [array_type]\n", argv[0]);
        printf("Array types: 0=sorted, 1=reverse sorted, 2=random/unsorted\n");
        printf("Using default values: 1000000000 1000000000 0\n\n");
    }
    
    // Check if requested sizes exceed the maximum
    if (m > max_array_size || n > max_array_size) {
        printf("Warning: Requested array sizes exceed available memory\n");
        printf("Adjusting array sizes to maximum possible: %lld\n", max_array_size);
        m = n = max_array_size;
    }
    
    printf("Initializing arrays of size %lld and %lld...\n", m, n);
    
    // Allocate memory for input arrays
    printf("Allocating memory...\n");
    int *A = (int *)malloc(m * sizeof(int));
    int *B = (int *)malloc(n * sizeof(int));
    
    if (!A || !B) {
        printf("Memory allocation for input arrays failed. Try smaller array sizes.\n");
        if (A) free(A);
        if (B) free(B);
        return 1;
    }
    
    // Define array types to test
    const char* array_type_names[] = {"SORTED", "REVERSE_SORTED", "RANDOM/UNSORTED"};
    ArrayType type = (ArrayType)array_type;
    
    // Generate the requested type of arrays
    printf("Generating %s arrays...\n", array_type_names[type]);
    generate_arrays(A, m, B, n, type);
    
    // Run benchmarks
    printf("\nRunning benchmarks...\n");
    run_benchmarks(A, m, B, n, array_type_names[type]);
    
    // Free allocated memory
    free(A);
    free(B);
    
    return 0;
}
