#include "heat_simulation.h"
#include <algorithm>

using namespace std;

#define TILE 32
#define CEIL_DIV(a,b) (((a) + (b) - 1) / (b))

const float EPS = 1e-3;  
const float diffusion_rate = 0.25f;

int grid_size = 0, block_size = 0;
float *grid = NULL;
float *new_grid = NULL;
size_t totalBlocks; 

// diffusion variables
dim3 block_diffuse, grid_diffuse;
size_t bytes_shared_diffusion = 0; 

// reduction variables
dim3 block_reduce, grid_reduce;
size_t bytes_shared_reduction = 0;

#define CUDA_CHECK(x) do {                                \
    cudaError_t err = (x);                                \
    if (err != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error %s at %s:%d\n",       \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                          \
    }                                                     \
} while (0)

__host__ void debug(float * gr, int n) {
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) printf("%4.0f", gr[y*n + x]);
        printf("\n");
    }
}

// TO-DO: implement graphics?

// Kernels (device) //////////////////////////////////////////////////
__global__ void heat_sources(float * grid, int grid_size) {
    int cx = grid_size / 2;
    int cy = grid_size / 2;

    grid[cy * grid_size + cx] = 100.0f;

    int offset = 20;
    grid[(cy + offset) * grid_size + (cx + offset)] = 100.0f;
    grid[(cy + offset) * grid_size + (cx - offset)] = 100.0f;
    grid[(cy - offset) * grid_size + (cx + offset)] = 100.0f;
    grid[(cy - offset) * grid_size + (cx - offset)] = 100.0f;
}

__global__ void grid_diffusion(float * grid, float * new_grid, int grid_size, float diffusion_rate) {
    extern __shared__ float shmem[];
    float *d_shared_grid = shmem;
    float *d_shared_grid_new = shmem + (blockDim.x +2) * (blockDim.y +2); 
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x == 0 || x >= grid_size-1 || y == 0 || y >= grid_size-1) {
        // bordes sin cambiar
        __syncthreads();
        __syncthreads();
        return;
    }
    
    // 1) Acceder coalescentemente a la memmoria global trayendo datos a memoria compartida
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

__global__ void grid_reduction(float * grid) {
    extern __shared__ float d_shared_grid[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + tid;
    d_shared_grid[tid] = grid[i] + grid[i+blockDim.x];

    __syncthreads();
    for (unsigned int jmp = blockDim.x/2; jmp > 0; jmp >>= 1) {
        if (tid < jmp) d_shared_grid[tid] += d_shared_grid[tid + jmp];
        __syncthreads();
    }
    if (tid == 0) grid[blockIdx.x] = d_shared_grid[0];
}


// General functions (host)  //////////////////////////////////////////////////
__host__ void initialize_grid(int N, int cuda_block_size) {
    grid_size = N;
    block_size = cuda_block_size; 
    
    // Diffusion
    block_diffuse = dim3(TILE, CEIL_DIV(block_size, TILE)); 
    grid_diffuse = dim3(CEIL_DIV(grid_size, block_diffuse.x), CEIL_DIV(grid_size, block_diffuse.y));
    bytes_shared_diffusion = sizeof(float) * (block_diffuse.x +2) * (block_diffuse.y +2) * 2; // *2 because we have 2 shared grids
    totalBlocks = (grid_diffuse.x * grid_diffuse.y); 

    // Reduction
    block_reduce = dim3(block_size, 1);
    grid_reduce = dim3(CEIL_DIV(N * N, block_size * 2), 1);
    bytes_shared_reduction = block_size * sizeof(float);

    size_t totalBytes = sizeof(float) * grid_size * grid_size;
    cudaMalloc(&grid, totalBytes);
    cudaMalloc(&new_grid, totalBytes);
    cudaMemset(grid, 0, totalBytes);
    cudaMemset(new_grid, 0, totalBytes);

    // apply heat sources in global GPU memory
    heat_sources<<<1, 1>>>(grid, grid_size);
}

__host__ bool update_simulation() {
    // calls kernels, reduces grid, and checks if end of simulation 

    // calculate the diffusion using shared GPU memory ////////////////////////////////////////////////////////
    grid_diffusion<<<grid_diffuse, block_diffuse, bytes_shared_diffusion>>>(grid, new_grid, grid_size, diffusion_rate);
    cudaDeviceSynchronize();
    swap(grid, new_grid);
    
    // apply heat sources in global GPU memory ////////////////////////////////////////////////////////////////
    heat_sources<<<1, 1>>>(grid, grid_size);

    // reduce in device using kernel //////////////////////////////////////////////////////////////////////////
    cudaMemcpy(new_grid, grid, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToDevice); // using new_grid as reduction vector 
    grid_reduce.x = CEIL_DIV(grid_size * grid_size, block_size * 2);
    while (grid_reduce.x > 1) {
        grid_reduction<<<grid_reduce, block_reduce, bytes_shared_reduction>>>(new_grid);
        cudaDeviceSynchronize();
        grid_reduce.x = CEIL_DIV(grid_reduce.x, block_size * 2);
    } 
    grid_reduction<<<grid_reduce, block_reduce, bytes_shared_reduction>>>(new_grid);
    cudaDeviceSynchronize();

    // bring reduction and calculate delta to check end of simulation //////////////////////////////////////////
    float reduction = 0.0; 
    static float prev_reduction = 0.0; 
    cudaMemcpy(&reduction, new_grid, sizeof(float), cudaMemcpyDeviceToHost);
    float delta = reduction - prev_reduction;
    prev_reduction = reduction; 
    
    return delta < EPS; 
}

__host__ void destroy_grid() {
    cudaFree(grid);
    cudaFree(new_grid);
    grid = NULL;
    grid_size = 0;
}
