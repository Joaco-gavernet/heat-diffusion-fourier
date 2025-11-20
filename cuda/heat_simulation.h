#ifndef HEAT_SIMULATION_H
#define HEAT_SIMULATION_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include <assert.h>   // o <cassert>
#include <stdio.h>    // para printf

void initialize_grid(int N, int cuda_block_size);
void update_simulation();
void destroy_grid();

#endif

