
#include "quadtree.h"


int get_quad(Vec pos, float bound[4]);

int set_bound_get_quad(Vec pos, float bound[4]);

QTree* init_qtree(float size){
	QTree *qtree = (QTree*) calloc(1, sizeof(QTree));
	qtree->arr = (Planet*) calloc(1,sizeof(Planet));
	qtree->arr_size = 1;
	qtree->tree_size = 0;
	qtree->size = size;
	qtree->arr[0].mass = -1;
	return qtree;
}

QTree *clear_qtree(QTree *qtree){
	int size = qtree->size;
	free(qtree->arr);
	free(qtree);
	return init_qtree(size);
}
QTree *clear_qtree(QTree *qtree, int size){
	if(size < 500){
		size = 500;
	}
	free(qtree->arr);
	free(qtree);
	return init_qtree(size);
}

void construct_qtree(QTree *qtree, Planet *planets, int size){
	for(int i = 0; i < size; i++){
		insert_planet(qtree, planets + i);
	}

}

void insert_planet(QTree *qtree, Planet *planet){
	bool inserted = false;
	int curr = 0;
	float size = (float)qtree->size;
	// up down r l
	float bound[4] = {size,-size,size,-size};
	Vec pos = planet->pos;
	Planet temp;
	static int maxlevel = 0;
	int level = 0;
	if(abs(pos.x)> size || abs(pos.y) > size){
		return;
	}

	int index;
	while(!inserted){
		if(	qtree->arr[curr].mass == -1 ||
			qtree->arr[curr].mass == 0){
			qtree->arr[curr].pos = planet->pos;
			qtree->arr[curr].dir = planet->dir;
			qtree->arr[curr].r = planet->r;
			qtree->arr[curr].mass = planet->mass;
			inserted = true;
		}else if(qtree->arr[curr].mass > 0){
			if((curr + 1) * 4 >= qtree->arr_size){
				qtree->arr = (Planet*) realloc(qtree->arr,
						((qtree->arr_size + (qtree->arr_size * 4))) * sizeof(Planet));
				memset(qtree->arr + qtree->arr_size,
						0,qtree->arr_size * 4 * sizeof(Planet));
				qtree->arr_size += qtree->arr_size * 4;
			}

			temp = qtree->arr[curr];
			float dist = sqrt(pow(temp.pos.y - planet->pos.y,2) + pow(temp.pos.x - planet->pos.x,2));
			if(dist < temp.r + planet->r){
				if(temp.mass > qtree->arr[curr].mass){
					qtree->arr[curr].pos.x = planet->pos.x;
					qtree->arr[curr].pos.y = planet->pos.y;
				}else{
			
				}
				float mass1 = planet->mass;
				float mass2 = qtree->arr[curr].mass;				
				float dirx = qtree->arr[curr].dir.x * mass1 +
						      planet->dir.x * mass2;
				float diry = qtree->arr[curr].dir.y * mass1+
						      planet->dir.y * mass2;
				qtree->arr[curr].dir.x = dirx / (mass2 + mass1);
				qtree->arr[curr].dir.y = diry / (mass2 + mass1);

				qtree->arr[curr].mass = planet->mass + temp.mass;
				qtree->arr[curr].r = log2f(qtree->arr[curr].mass) * 0.20;
				inserted = true;			
			}else{
				qtree->arr[curr].mass = -2;
				index = 4 * curr + 1 + get_quad(temp.pos,bound);
				qtree->arr[index] = temp;
			}

		}
		if(level > 13){
			inserted = true;
			printf("tree to deep!\n");
		}
		level++;
		curr = 4 * curr + 1 + set_bound_get_quad(pos, bound); 

		
	}
	if(level > maxlevel){
		printf("maxlevel:%d\n",maxlevel);
		maxlevel = level;
	}
	qtree->tree_size ++;

}

int get_quad(Vec pos, float bound[4]){
		if(pos.x >= (bound[3] + bound[2])/2.0 && 
		   pos.y >= (bound[0] + bound[1])/2.0){
			return 0;
		}
		if(pos.x < (bound[3] + bound[2])/2.0 && 
		   pos.y >= (bound[0] + bound[1])/2.0){
			return 1;
		}
		if(pos.x >= (bound[3] + bound[2])/2.0 && 
		   pos.y < (bound[0] + bound[1])/2.0){
			return 2;
		}
		if(pos.x < (bound[3] + bound[2])/2.0 && 
		   pos.y < (bound[0] + bound[1])/2.0){
			return 3;
		}
		return 0;
}

int collaps_tree(QTree *qtree, PlanetsArr* container){
	int j = 0;
	for(int i = 0; i < qtree->arr_size; i++){
		if(qtree->arr[i].mass > 0){
			container->planets[j] = qtree->arr[i];
			j++;
		}
	
	}
	int delta  =container->size_arr-1 - j;
	if(delta >= PLANET_BLOCK_N){ 
		container->size_arr = container->size_arr - (delta -delta%PLANET_BLOCK_N);		
		container->planets = (Planet*) realloc(container->planets,container->size_arr * sizeof(Planet));

	}
	
	qtree->tree_size = 0;
	return j;
}


int set_bound_get_quad(Vec pos, float bound[4]){
	if(pos.x >= (bound[3] + bound[2])/2.0 && 
		   pos.y >= (bound[0] + bound[1])/2.0){
			bound[3] = (bound[3] + bound[2])/2.0; 
			bound[1] = (bound[0] + bound[1])/2.0; 
			return 0;
		}else if(pos.x < (bound[3] + bound[2])/2.0 && 
		   pos.y >= (bound[0] + bound[1])/2.0){
			bound[2] = (bound[3] + bound[2])/2.0; 
			bound[1] = (bound[0] + bound[1])/2.0; 
			return 1;
		}else if(pos.x >= (bound[3] + bound[2])/2.0 && 
		   pos.y < (bound[0] + bound[1])/2.0){
			bound[3] = (bound[3] + bound[2])/2.0; 
			bound[0] = (bound[0] + bound[1])/2.0; 
			return 2;
		}else if(pos.x < (bound[3] + bound[2])/2.0 && 
		   pos.y < (bound[0] + bound[1])/2.0){
			bound[2] = (bound[3] + bound[2])/2.0; 
			bound[0] = (bound[0] + bound[1])/2.0;
			return 3;
		}
	return 0;
}

