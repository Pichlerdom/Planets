#ifndef PLANETS_H_
#define PLANETS_H_

#include "includes.h"
typedef struct{
	float x,y;
}Vec;

typedef struct{
	Vec pos;
	Vec dir;
	float mass;
	float r;
}Planet;

typedef struct{
	Planet *planets;
	pthread_mutex_t planetsMutex;
	int size_arr;
	int number;
	bool quit;
}PlanetsArr;

typedef struct {
	struct{
		char* name;
		struct{
			unsigned int height;
			unsigned int width;
		}dim;
	}screen;
	struct{
		float speed;
		float r;
		float mass_max;
		float mass_min;
	}planets;
}PConfig;

typedef struct{
	PlanetsArr *container;
	PConfig *pconfig;
}ThreadArgs;

#include "planets_config.h"
#include "planets_display.h"
#include "planets_func.h"
#include "planets_cuda_kernel.h"
#include "planets_cuda_func.h"

#endif
