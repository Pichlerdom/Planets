#include "quadtree.h"

#define QTREE_ARR_SIZE 244140625
#define QTREE_MAX_LEVEL 13


int get_quad(Vec pos, float bound[4]);

int set_bound_get_quad(Vec pos, float bound[4]);

QTree* init_qtree(float size){
	QTree *qtree = (QTree*) calloc(32, sizeof(QTree));
	qtree->arr = (QTree_Node*) calloc(QTREE_ARR_SIZE, sizeof(32));

	qtree->number_of_nodes = 0;
	qtree->arr_size = 32;
	qtree->size = size;
	qtree->max_level = QTREE_MAX_LEVEL;

	init_QTree_Node(qtree, 0);
	return qtree;
}

void clear_qtree(QTree *qtree){
	free(qtree->arr);
	free(qtree);
}


int construct_qtree(QTree *qtree, Planet *planets, int size){
	int inserted = 0; 
	for(int i = 0; i < size; i++){
		if(insert_planet(qtree, planets, i)){
			inserted++;
		}
	}
}

bool insert_planet(QTree *qtree, Planet *planets, int index){
	bool inserted = false;
	int curr = 0;
	float size = (float)qtree->size;
	// up down r l
	float bound[4] = {size,-size,size,-size};
	Vec pos = planets[index].pos;
	QTree_Node *nodes = qtree->arr;

	if(planets[index].mass < -1){
		planets[index].mass = -1;
		return false;
	}
	if(abs(pos.x)> size || abs(pos.y) > size){
		planets[index].mass = -1;
		return false;
	}
//	printf("number_of_nodes: %d\n", qtree->number_of_nodes);
	while(!inserted){
		if(!nodes[curr].is_split){
			if(nodes[curr].number >= __QTREE_PLANETS_PER_QUAD){
				split_node(qtree, planets,bound, curr);
				nodes[curr].is_split = true;

			}else{
				int temp;
				int dist;
				for(int i = 0; i < nodes[curr].number; i++){
					temp = nodes[curr].inside[i];
					dist = sqrt(pow(planets[temp].pos.x - pos.x,2.0) + pow(planets[temp].pos.y - pos.y,2.0));
					int curr_r = planets[temp].r;
					int temp_r = planets[index].r;
				
					if(dist < curr_r + temp_r){
						float mass1 = planets[temp].mass;
						float mass2 = planets[index].mass;	

						if(mass1 > mass2)
						{
							planets[index].pos.x = planets[temp].pos.x;
							planets[index].pos.y = planets[temp].pos.y;
						}
						nodes[curr].inside[i] = index;

								
						float dirx = planets[index].dir.x * mass2 +
									 planets[temp].dir.x * mass1;
									
						float diry = planets[index].dir.y * mass2 +
									 planets[temp].dir.y * mass1;

						planets[index].dir.x = dirx / (mass2 + mass1);
						planets[index].dir.y = diry / (mass2 + mass1);
						planets[index].mass = planets[index].mass + planets[temp].mass;

				    	planets[index].r =  (sqrt(planets[index].mass))/100;//log2f(planets[index].mass) * 0.18;;
						planets[temp].mass = -1;
					
						inserted = true;
						return false;		
					}
				
				}
				nodes[curr].inside[nodes[curr].number] = index;
				nodes[curr].number++;
			
				if(nodes[curr].number > __QTREE_PLANETS_PER_QUAD){
					nodes[curr].number = __QTREE_PLANETS_PER_QUAD;
				}
				inserted = true;
			}
		}else{
			int quad = set_bound_get_quad(pos, bound);
			switch(quad){
				case 0:
					curr = nodes[curr].nw;
					break;
				case 1:
					curr = nodes[curr].ne;
					break;
				case 2:
					curr = nodes[curr].sw;
					break;
				case 3:
					curr = nodes[curr].se;
					break;
			}
		}
	}
	
	return true;
}

void split_node(QTree *qtree, Planet *planets, float bound[4], int node_index){
	if(qtree->number_of_nodes + 4 >= qtree->arr_size){
		qtree->arr = (QTree_Node*) realloc(qtree->arr, (qtree->arr_size + 4) * sizeof(QTree_Node));
		qtree->arr_size = qtree->arr_size + 4;
	}
	
	QTree_Node *nodes = qtree->arr;
	int number_of_nodes = qtree->number_of_nodes;
	for(int i = 1; i <= 4; i++){
		init_QTree_Node(qtree, number_of_nodes + i);
	}
	for(int i = 0; i < __QTREE_PLANETS_PER_QUAD; i++){
			Vec pos = planets[nodes[node_index].inside[i]].pos;
			int quad = get_quad(pos, bound) + 1;
			int index = number_of_nodes + quad;
			
			nodes[index].inside[nodes[index].number] = nodes[node_index].inside[i];
			nodes[index].number++;
	}
	
	nodes[node_index].nw = number_of_nodes + 1;
	nodes[node_index].ne = number_of_nodes + 2;
	nodes[node_index].sw = number_of_nodes + 3;
	nodes[node_index].se = number_of_nodes + 4;


	qtree->number_of_nodes = number_of_nodes + 4;
}

void init_QTree_Node(QTree *qtree, int index){
	qtree->arr[index].is_split = false;
	qtree->arr[index].nw = -1; 
	qtree->arr[index].ne = -1; 
	qtree->arr[index].sw = -1; 
	qtree->arr[index].se = -1; 
	qtree->number_of_nodes = 0;
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

int collaps_tree(QTree *qtree, Planet* planets,Planet* planets_out,int curr, int index){
	if(qtree->arr[curr].is_split){

		index = collaps_tree(qtree,planets,planets_out,qtree->arr[curr].ne, index);
		index = collaps_tree(qtree,planets,planets_out,qtree->arr[curr].nw, index);
		index = collaps_tree(qtree,planets,planets_out,qtree->arr[curr].se, index);
		index = collaps_tree(qtree,planets,planets_out,qtree->arr[curr].sw, index);

	}else{
		int j = 0;
		for(int i = 0; i < qtree->arr[curr].number; i++){
			j  = qtree->arr[curr].inside[i];
			planets_out[index + i] = planets[j];
		}
		index = index + qtree->arr[curr].number;
	}
	return index;
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

