
#include "planets.h"

Display* init_display(PConfig *pconfig){ 
	Display *display = (Display *) calloc(1,sizeof(Display));
	if(display == NULL){
		printf("Display could not be created!");
		return NULL;
	}
	//Initialize SDL
	if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	{
		printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
	    
        return NULL;
	}
    display->window = SDL_CreateWindow(pconfig->screen.name,
                                       SDL_WINDOWPOS_UNDEFINED,
                                       SDL_WINDOWPOS_UNDEFINED,
                                       pconfig->screen.dim.width,
                                       pconfig->screen.dim.height,
                                       SDL_WINDOW_SHOWN);
    if(display->window == NULL) {
        printf("Unable to create Window! SDL_Error: %s\n",SDL_GetError());
        return NULL;
    }
	display->renderer = SDL_CreateRenderer( display->window, -1, SDL_RENDERER_ACCELERATED );
	if( display->renderer == NULL )
	{
		printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
		return NULL;
	}

	//Initialize renderer color
	SDL_SetRenderDrawColor(display->renderer, 0xFF, 0xFF, 0xFF, 0xFF ); 
	
	display->pos.x = 0.0;
	display->pos.y = 0.0;
    return display;
 
}



void* main_display_loop(void* arguments){
	//Display contains all the structures we need for rendering stuff
	ThreadArgs *args = (ThreadArgs*) arguments;
	PlanetsArr *container = args->container;
	PConfig *pconfig = args->pconfig;
	
	Display *pdisplay = init_display(pconfig);

	SDL_Event e;
	uint32_t currTime = SDL_GetTicks();
	uint32_t frameTime = 0u;


	
	while(!container->quit){
		currTime = SDL_GetTicks();		
		
		//event handling loop
		while(SDL_PollEvent(&e) != 0){

			switch(e.type){
			
				//close application
				case SDL_QUIT:
					container->quit = true;
				break;
				//mouse wheel events
				case SDL_MOUSEWHEEL:	
					if(e.wheel.y == 1){
						pconfig->scale *= 1.1;
					}else if(e.wheel.y == -1){
						pconfig->scale /= 1.1;
					}
				break;
				case SDL_MOUSEBUTTONUP:
					if(e.button.type == SDL_BUTTON_LEFT){
						pdisplay->pos.x = e.button.x;
						pdisplay->pos.y = e.button.y;
					}
				break;
			}
		}
		
	pdisplay->pos.y = container->planets[find_biggest_mass(container)].pos.y * pconfig->scale - pconfig->screen.dim.height/2;
	pdisplay->pos.x = container->planets[find_biggest_mass(container)].pos.x * pconfig->scale- pconfig->screen.dim.width/2;

		//Clear screen
		SDL_SetRenderDrawColor( pdisplay->renderer, 0x00, 0x00, 0x00, 0xFF );
		SDL_RenderClear( pdisplay->renderer );
		draw_planets(pdisplay, container, pconfig->scale);
		//Update screen
		SDL_RenderPresent( pdisplay->renderer );

		//FPS stuff
		frameTime = SDL_GetTicks() - currTime;
		
		//printf("disp:%d\n", frameTime);

		if(frameTime > MS_PER_FRAME){
			frameTime = MS_PER_FRAME;
		}
		SDL_Delay(MS_PER_FRAME-frameTime);

	}

	printf("Display thread exited:\n");

	close(pdisplay);
	return NULL;

}


void draw_planets(Display *display, PlanetsArr *container, float scale){
	pthread_mutex_lock(&container->planetsMutex);
	for(int i = 0; i <  container->number; i++){
	
		Vec pos = container->planets[i].pos;
		pos.x *= scale;
		pos.y *= scale;
		Vec dir = container->planets[i].dir;
		dir.x *= __MOVE * 10;
		dir.y *= __MOVE * 10;
		float r = container->planets[i].r;
		int blockn = (int) (i / PLANET_BLOCK_N) + 10;
		
		SDL_SetRenderDrawColor(display->renderer, blockn * 1000, blockn * 10, blockn * 100, 0xFF);
		
		draw_planet(display, &pos, r * scale);
		
		if(container->planets[i].mass > 0){ 
			SDL_SetRenderDrawColor(display->renderer, 0xFF, 0x00, 0xFF, 0x55);
			SDL_RenderDrawLine(display->renderer,
								pos.x, pos.y,
								pos.x + dir.x, pos.y + dir.y);
		}
	}
	pthread_mutex_unlock(&container->planetsMutex);

	
}


void draw_planet(Display *display, Vec *pos, float r){
	int radius = r;
	pos->x -= display->pos.x;
	pos->y -= display->pos.y;
	
	SDL_RenderDrawPoint(display->renderer, pos->x, pos->y);

	for (int w = 0; w < radius * 2; w++)
	{
		for (int h = 0; h < radius * 2; h++)
		{
			int dx = radius - w; // horizontal offset
			int dy = radius - h; // vertical offset
			if ((dx*dx + dy*dy) <= (radius * radius))
			{
				SDL_RenderDrawPoint(display->renderer, pos->x + dx, pos->y + dy);
			}
		}
	}
}


void close(Display* display){
    
	SDL_DestroyRenderer(display->renderer);
	SDL_DestroyWindow(display->window);
	//Quit SDL subsystems
	SDL_Quit(); 

	free(display);
}



