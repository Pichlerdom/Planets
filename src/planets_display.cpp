
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
	display->renderer = SDL_CreateRenderer( display->window, -1, SDL_RENDERER_SOFTWARE );
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
	
	TTF_Init();
	TTF_Font *font = TTF_OpenFont("RobotoMono-Medium.ttf", 12);
	SDL_Texture *texture;
	SDL_Rect textrect;

	char *tempstr = (char*) calloc(256,sizeof(char));
	
	Vec screenpos;

	
	screenpos.y = container->planets[find_biggest_mass(container)].pos.y * pconfig->scale - pconfig->screen.dim.height/2;
	screenpos.x = container->planets[find_biggest_mass(container)].pos.x * pconfig->scale- pconfig->screen.dim.width/2;

	int mousex = 0;
int mousey = 0;
int retval;
	while(!container->quit){
		currTime = SDL_GetTicks();		
		
		//event handling loop
		while(SDL_PollEvent(&e) != 0){

			SDL_GetMouseState(&mousex, &mousey);
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
					screenpos.x += (mousex / pconfig->scale) - pdisplay->pos.x - DEFAULT_SCREEN_WIDTH/2;
					screenpos.y += (mousey / pconfig->scale) - pdisplay->pos.y  - DEFAULT_SCREEN_HEIGHT/2;
				break;
			}
		}

		
		pdisplay->pos.y = (screenpos.y* pconfig->scale  - pconfig->screen.dim.height/2);
		pdisplay->pos.x = (screenpos.x* pconfig->scale - pconfig->screen.dim.width/2);


		//Clear screen
		SDL_SetRenderDrawColor( pdisplay->renderer, 0x00, 0x00, 0x00, 0xFF );
		SDL_RenderClear( pdisplay->renderer);
		
	
		retval = pthread_mutex_lock(&(args->qtreeMutex));
		printf("%d\n",args->qtree->arr_size);
		
		if(args->qtree->arr_size > 0){

			float bound[4] = {args->qtree->size,-args->qtree->size,args->qtree->size,-args->qtree->size};
			draw_QTree(pdisplay, args->qtree, pconfig->scale, bound,0);

		}	
		
		pthread_mutex_unlock(&(args->qtreeMutex));

		draw_planets(pdisplay, container, pconfig->scale);
		
		SDL_SetRenderDrawColor( pdisplay->renderer, 0xff, 0xff, 0xff, 0xFF );
		sprintf(tempstr,"FPS:%.2f",1000.0/args->calctime);
		get_text_and_rect(pdisplay->renderer,
						  10, 10,
						  tempstr,
		font, &texture, &textrect);
		SDL_RenderCopy(pdisplay->renderer, texture, NULL, &textrect);
		
		SDL_SetRenderDrawColor( pdisplay->renderer, 0xff, 0xff, 0xff, 0xFF );
		sprintf(tempstr,"Number:%d",  container->number);
		get_text_and_rect(pdisplay->renderer,
						  10, 30,
						  tempstr,
		font, &texture, &textrect);
		SDL_RenderCopy(pdisplay->renderer, texture, NULL, &textrect);
		
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
/*
- x, y: upper left corner.
- texture, rect: outputs.
*/
void get_text_and_rect(	SDL_Renderer *renderer, 
						int x, int y,
						char *text,
						TTF_Font *font, SDL_Texture **texture, SDL_Rect *rect) {
    int text_width;
    int text_height;
    SDL_Surface *surface;
    SDL_Color textColor = {255, 255, 255, 0};

    surface = TTF_RenderText_Solid(font, text, textColor);
    *texture = SDL_CreateTextureFromSurface(renderer, surface);
    text_width = surface->w;
    text_height = surface->h;
    SDL_FreeSurface(surface);
    rect->x = x;
    rect->y = y;
    rect->w = text_width;
    rect->h = text_height;
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
void draw_QTree(Display *display, QTree *qtree ,float scale, float bound[4], int curr){
	if(qtree->arr[curr].mass == -1 ||
		qtree->arr[curr].mass == 0 ){
		
		SDL_SetRenderDrawColor(display->renderer, 0x00, 0xff, 0xff, 0xff);
	}else if (qtree->arr[curr].mass == -2){
		float bounds[4];
		bounds[0] = bound[0];
		bounds[1] = (bound[0] + bound[1])/2.0;
		bounds[2] = bound[2];
		bounds[3] = (bound[3] + bound[2])/2.0; 
 
		draw_QTree(display,qtree, scale,bounds, curr * 4 + 1);
		
		bounds[0] = bound[0];
		bounds[1] = (bound[0] + bound[1])/2.0;
		bounds[2] = (bound[3] + bound[2])/2.0;
		bounds[3] = bound[3];  
		draw_QTree(display,qtree, scale,bounds, curr * 4 + 2);
		
		
		bounds[0] = (bound[0] + bound[1])/2.0;
		bounds[1] = bound[1];
		bounds[2] = bound[2];
		bounds[3] = (bound[3] + bound[2])/2.0;  
		draw_QTree(display,qtree,scale, bounds, curr * 4 + 3);
		
		bounds[0] = (bound[0] + bound[1])/2.0; 
		bounds[1] = bound[1];
		bounds[2] = (bound[3] + bound[2])/2.0;
		bounds[3] = bound[3]; 
		draw_QTree(display,qtree,scale,bounds, curr * 4 + 4);

		SDL_SetRenderDrawColor(display->renderer, 0xff, 0x00, 0xff, 0xff);


	} else {
		SDL_SetRenderDrawColor(display->renderer, 0x00, 0xff, 0x00, 0xff);
	}
	SDL_Rect rect;
	rect.x = bound[3] * scale - display->pos.x;  
	rect.y = bound[0] * scale - display->pos.y;
	rect.w = (bound[2] - bound[3]) * scale;
	rect.h = (bound[1] - bound[0]) * scale;
	SDL_RenderDrawRect(display->renderer, &rect);
}

void close(Display* display){
    
	SDL_DestroyRenderer(display->renderer);
	SDL_DestroyWindow(display->window);
	//Quit SDL subsystems
	SDL_Quit(); 

	free(display);
}



