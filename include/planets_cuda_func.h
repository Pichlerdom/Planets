#ifndef _PLANETS_CUDA_FUNC_H_
#define _PLANETS_CUDA_FUNC_H_


#define TICKS 60
#define MS_PER_TICK 1000/TICKS


void move_planets(PlanetsArr* container);

void remove_dead_planets(PlanetsArr* container);

void realloc_cuda(PlanetsArr* container, GPU_Mem *gpu_mem);

void *main_calc_loop(void* cont);
#endif
