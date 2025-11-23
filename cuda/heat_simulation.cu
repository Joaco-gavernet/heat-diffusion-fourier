#include "heat_simulation.h"
#include <algorithm>

using namespace std;

#define TILE 32
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

dim3 h_blockDim, h_gridDim;
size_t bytesShared; 
float diffusion_rate = 0.25f;

float *grid = NULL;
float *new_grid = NULL;
float *partial = NULL;
int grid_size = 0;

#define CUDA_CHECK(x) do {                                \
    cudaError_t err = (x);                                \
    if (err != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error %s at %s:%d\n",       \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                          \
    }                                                     \
} while (0)

// general TO-DOs: 
// [x] fix consistency for first single step (should total = 1000 but sometimes differs)
// [x] compare against sequential solution
// [x] clean code
// [.] Use partial vector to reduce grid
// [.] Add convergence verification to stop simulation
// [.] check if graphics integration works ok


// Kernels (device) //////////////////////////////////////////////////
__global__ void heat_sources(float * grid, int grid_size) {
    int cx = grid_size / 2 +2;
    int cy = grid_size / 2 +2;
    // printf("(device) inside heat_sources, center (y,x): (%d,%d)\n", cy, cx);

    grid[cy * grid_size + cx] = 100.0f;

    int offset = 20;
    grid[(cy + offset) * grid_size + (cx + offset)] = 100.0f;
    grid[(cy + offset) * grid_size + (cx - offset)] = 100.0f;
    grid[(cy - offset) * grid_size + (cx + offset)] = 100.0f;
    grid[(cy - offset) * grid_size + (cx - offset)] = 100.0f;
}

__global__ void simulate_diffusion(float * grid, float * new_grid, int grid_size, float diffusion_rate) {
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && threadIdx.y == 1) printf("(device) inside simulate_diffusion\n");
    extern __shared__ float shmem[];
    float *d_shared_grid = shmem;
    float *d_shared_grid_new = shmem + (blockDim.x +2) * (blockDim.y +2); 
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // TO-DO: check if logic is ok to avoid processing borders, but still syncing with the rest of threads
    if (x == 0 || x >= grid_size-1 || y == 0 || y >= grid_size-1) {
        // d_shared_grid[threadIdx.y *blockDim.x + threadIdx.x] = grid[y*grid_size + x]; // bordes sin cambiar
        __syncthreads();
        __syncthreads();
        return;
    }
    
    // 1) Acceder coalescentemente a la memmoria global trayendo datos a memoria compartida
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 1 && threadIdx.y == 1) printf("(device) coalescent access\n");
    
    const int strideY = blockDim.x + 2;
    const int localX = threadIdx.x + 1;
    const int localY = threadIdx.y + 1;

    d_shared_grid[localY * strideY + localX] = grid[y*grid_size + x];
    if (threadIdx.y == 0) d_shared_grid[localX] = grid[(y -1)*grid_size + x];
    else if (threadIdx.y == blockDim.y - 1) d_shared_grid[(localY + 1) * strideY + localX] = grid[(y +1)*grid_size + x];
    if (threadIdx.x == 0) d_shared_grid[localY * strideY] = grid[y*grid_size + x - 1];
    else if (threadIdx.x == blockDim.x - 1) d_shared_grid[localY * strideY + (localX + 1)] = grid[y*grid_size + x + 1];

    __syncthreads();
    
    // 2) Procesamiento sobre memoria compartida
    float center = d_shared_grid[localY * strideY + localX];

    float up = d_shared_grid[(localY - 1) * strideY + localX];
    float down = d_shared_grid[(localY + 1) * strideY + localX];
    float left = d_shared_grid[localY * strideY + (localX - 1)];
    float right = d_shared_grid[localY * strideY + (localX + 1)];
    d_shared_grid_new[localY *strideY + localX] = center + diffusion_rate * (up + down + left + right - 4.0f * center);
    
    __syncthreads();
    
    // 3) Acceder coalescentemente a la memoria global escribiendo resultados
    new_grid[y*grid_size + x] = d_shared_grid_new[localY *strideY + localX];
}

__global__ void reduce_grid(float * grid, float * partial, int n) {
    extern __shared__ float d_shared_grid[];
    
    const unsigned int threads_per_block = blockDim.x * blockDim.y;
    const unsigned int local_id = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int global_id = block_id * threads_per_block + local_id;

    d_shared_grid[local_id] = (global_id < n) ? grid[global_id] : 0.0f;
    __syncthreads();

    for (unsigned int s = threads_per_block >> 1; s > 0; s >>= 1) {
        if (local_id < s) d_shared_grid[local_id] += d_shared_grid[local_id + s];
        __syncthreads();
    }
    
    if (local_id == 0) partial[block_id] = d_shared_grid[0];
}


// General functions (host)  //////////////////////////////////////////////////
void initialize_grid(int N, int cuda_block_size) {
    // TO-DO: cudaGetDeviceProperties() and checkparams(N, cuda_block_size);

    grid_size = N;
    
    assert(cuda_block_size % 32 == 0); // TO-DO: check if this assumption is correct
    h_blockDim = dim3(TILE, CEIL_DIV(cuda_block_size, TILE)); 
    h_gridDim = dim3(CEIL_DIV(grid_size, h_blockDim.x), CEIL_DIV(grid_size, h_blockDim.y));
    bytesShared = sizeof(float) * (h_blockDim.x +2) * (h_blockDim.y +2) * 2; // *2 because we have 2 shared grids

    size_t totalBytes = sizeof(float) * grid_size * grid_size;
    cudaMalloc(&grid, totalBytes);
    cudaMalloc(&new_grid, totalBytes);
    cudaMemset(grid, 0, totalBytes);
    cudaMemset(new_grid, 0, totalBytes);

    // apply heat sources in global GPU memory
    heat_sources<<<1, 1>>>(grid, grid_size);
    CUDA_CHECK(cudaGetLastError());
}

bool update_simulation() {
    // calls kernels, reduces grid, and checks if end of simulation 

    // calculate the diffusion using shared GPU memory ////////////////////////////////////////////////////////
    simulate_diffusion<<<h_gridDim, h_blockDim, bytesShared>>>(grid, new_grid, grid_size, diffusion_rate);
    cudaDeviceSynchronize();
    swap(grid, new_grid);
    
    // apply heat sources in global GPU memory ////////////////////////////////////////////////////////
    heat_sources<<<1, 1>>>(grid, grid_size);

    // reduce in device using kernel ////////////////////////////////////////////////////////
    size_t totalBlocks = (h_gridDim.x * h_gridDim.y); 
    cudaMemset(new_grid, 0, totalBlocks * sizeof(float)); // using new_grid as reduction vector 
    
    size_t nohalo = sizeof(float) * h_blockDim.x * h_blockDim.y;
    reduce_grid<<<h_gridDim, h_blockDim, nohalo>>>(grid, new_grid, grid_size * grid_size);
    
    return false; 
}

void destroy_grid() {
    cudaFree(grid);
    cudaFree(new_grid);
    grid = NULL;
    grid_size = 0;
}
