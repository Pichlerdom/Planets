#ifndef _QUADTREE_H_
#define _QUADTREE_H_

#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <SDL2/SDL.h>

#include "planets.h"


QTree* init_qtree(float size);

void construct_qtree(QTree *qtree, Planet *planets, int size);

void insert_planet(QTree *qtree, Planet *planets, int index);

//int collaps_tree(QTree *qtree, Planet* planets);

QTree *clear_qtree(QTree *qtree, int size);
void clear_qtree(QTree *qtree);
#endif

