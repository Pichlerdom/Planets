#ifndef _PLANETS_CUDA_FUNC_H_
#define _PLANETS_CUDA_FUNC_H_


#define TICKS 120
#define MS_PER_TICK 1000/TICKS

void move_planets(PlanetsArr* container);

void remove_dead_planets(PlanetsArr* container);

void *main_calc_loop(void* cont);
#endif
