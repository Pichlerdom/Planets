#ifndef _PLANETS_FUNC_H_
#define _PLANETS_FUNC_H_

#include "planets.h"
void create_random_planet(PlanetsArr *container, PConfig *pconfig);

PlanetsArr* init_planets();

void fill_planets(PlanetsArr* container, PConfig *pconfig);

int find_biggest_mass(PlanetsArr* container);

#endif
