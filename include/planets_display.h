#ifndef _PLANETS_DISPLAY_H_
#define _PLANETS_DISPLAY_H_


#include "planets.h"

#define FPS 60
#define MS_PER_FRAME 1000/FPS

typedef struct{
	SDL_Renderer* renderer;
	SDL_Window* window;
	Vec pos;	//screen position in world
}Display;

Display* init_display(PConfig *pconfig);

void* main_display_loop(void * cont);

void draw_planets(Display *display, PlanetsArr *container, float scale);

void draw_planet(Display *display, Vec *pos, float r);

void close(Display *display);

void draw_QTree(Display *display, QTree *qtree ,float scale, float bound[4], int curr);


void get_text_and_rect(	SDL_Renderer *renderer, 
						int x, int y,
						char *text,
						TTF_Font *font, SDL_Texture **texture, SDL_Rect *rect);
#endif
