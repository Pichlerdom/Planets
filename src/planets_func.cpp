#include "planets.h"

void create_random_planet(Planet *newPlanet, PConfig *pconfig){


	if(newPlanet == NULL){
		printf("Could not create random planet!\n");
	}
	int r = (int) pconfig->planet.r;
	float dist_from_center = (rand()%(r * 1000))/1000.0+__MIN_R;
	float alpha = (((rand()%(90*100))/100.0) * M_PI)/180.0;
	newPlanet->pos.x = (pconfig->screen.dim.width/2.0) + sinf(alpha) * dist_from_center * pow(-1,rand()%2);
	newPlanet->pos.y = (pconfig->screen.dim.height/2.0) + cosf(alpha) * dist_from_center * pow(-1,rand()%2);

	float dy = -((pconfig->screen.dim.width)/2 - newPlanet->pos.x);
	float dx = pconfig->screen.dim.height/2 - newPlanet->pos.y;
	
	float dist = sqrt(dy * dy + dx * dx)*__DIST;
	float speed = pconfig->planet.speed;
	
	newPlanet->dir.x = dx * (speed / (dist * dist));
	newPlanet->dir.y = dy * (speed / (dist * dist));

	int m_max = (int)pconfig->planet.mass_max;
	int m_min = (int)pconfig->planet.mass_min;
	newPlanet->mass = (rand()%((m_max-m_min)*1000))/1000.0+m_min;
	if(rand()%1000 == 0){
		newPlanet->mass *= 1000;
	}
	newPlanet->r = log2f(newPlanet->mass) * 0.18;
}

PlanetsArr* init_planets(){
	PlanetsArr* container = (PlanetsArr*) calloc(1, sizeof(PlanetsArr));
	if(container == NULL){
		printf("Could not allocate memory for the Planets data structur!\n");
		exit(-1);
	}

	//start out by making the array a fixed size
	container->planets = (Planet*) calloc(STARTARRSIZE, sizeof(Planet));
	if(container->planets == NULL){
		free(container);
		printf("Could not allocate memory for the Planets data structur!\n");
		exit(-1);
	}
	container->size_arr = STARTARRSIZE;
	container->number = 0;
	container->quit = false;

	return container;
}

void fill_planets(PlanetsArr* container, PConfig *pconfig){
	//seed random number generator
	srand(time(0));	
	
	pthread_mutex_lock(&container->planetsMutex);
	//point to the array that holdes all the planet structures
	container->planets[0].pos.x = pconfig->screen.dim.width/2;
	container->planets[0].pos.y = pconfig->screen.dim.height/2;
	container->planets[0].dir.x = 20;
	container->planets[0].dir.y = 0;

	container->planets[0].mass = __SEEDMASS;

	container->planets[0].r = log2f(container->planets[0].mass ) * 0.18;
	container->number++;

	for(int i = 1; i < NUMBER_OF_PLANETS; i++){
		create_random_planet(container->planets + i, pconfig);	
		container->number++;
	}

	pthread_mutex_unlock(&container->planetsMutex);
}

int find_biggest_mass(PlanetsArr* container){
	int biggest = 0;
	for(int i = 0; i < container->number; i++){
		if(container->planets[biggest].mass < container->planets[i].mass){
			biggest = i;
		}
	}
	return biggest;
}
