#ifndef _QUADTREE_H_
#define _QUADTREE_H_

#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <SDL2/SDL.h>

#include "planets.h"


QTree* init_qtree(float size);

int construct_qtree(QTree *qtree, Planet *planets, int size);

bool insert_planet(QTree *qtree, Planet *planets, int index);

void split_node(QTree *qtree, Planet *planets, float bound[4], int node_index);

void init_QTree_Node(QTree *qtree, int index);

int collaps_tree(QTree *qtree, Planet* planets,Planet* planets_out,int curr, int index);

QTree *clear_qtree(QTree *qtree, int size);
void clear_qtree(QTree *qtree);
#endif

