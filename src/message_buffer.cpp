/*include "planets.h"

typedef struct {
	int elem_num;
	int arr_size;
	Message_Buffer_Elem messages;
}Message_Buffer;

typedef struct {
	unsigned int lifespan; // lifespan in frames;
	char* message;
}Message_Buffer_Elem;


#define DEFAULT_MESSAGE_BUFFER_SIZE 16

void init_message_buffer(Message_Buffer *mbuffer){
	mbuffer = (Message_Buffer *) calloc(1, sizeof(Message_Buffer));
	if(mbuffer == NULL){
		printf("Error allocating Message Buffer!\n");
		return;
	}
	mbuffer->messages = (Message_Buffer_Elem*) calloc(DEFAULT_MESSAGE_BUFFER_SIZE, sizeof(Message_Buffer_Elem));
	if(mbuffer->messages == NULL){
		printf("Error allocating Message Buffer!\n");
		return;
	}
}

void resize_message_buffer(Message_Buffer *mbuffer){

	mbuffer->messages = (Message_Buffer_Elem*) calloc(mbuffer->arr_size * 2, sizeof(Message_Buffer_Elem));
	if(mbuffer->messages == NULL){
		printf("Error allocating Message Buffer!\n");
		return;
	}
	mbuffer->arr_size *= 2;
}

void add_message(Message_Buffer mbuffer, char *message, unsigned int lifespan){
	if(mbuffer->elem_num == mbuffer->arr->size){
		resize_message_buffer(mbuffer);
	}
	int m_len = strlen(message);
	char *m = (char *) calloc(m_len + 1,sizeof(char));
	strcpy(m,message);
	mbuffer->messages[mbuffer->elem_num].message = m;
	mbuffer->messages[mbuffer->elem_num].lifespan = lifespan;
	mbuffer->elem_num++;
}

char* get_message_at(Message_Buffer mbuffer, int index){
	return mbuffer->messages[index].message;
}

void update_message_buffer(Message_Buffer *mbuffer){
	for(int i = 0; i < mbuffer->elem_num; i++){
		if(mbuffer->messages[i].lifespan <= 0){
			remove_message_at(i);
		}else{
			mbuffer->message[i].lifespan--;
		}
	
	}
}


//shifts elements to the left
void remove_message_at(Message_Buffer mbuffer, int index){
	if(index >= mbuffer->elem_num){
		return;
	}
	free_message(mbuffer->messages[index]);
	for(int i = index; i < mbuffer->elem_num - 1; i++){
		mbuffer->message[i] = mbuffer->message[i + 1];
	}
	mbuffer->elem_num--;
}

void delete_message_buffer(Message_Buffer *mbuffer){
	free_message_buffer(mbuffer);
	free(mbuffer);
}


void free_message_buffer(Message_Buffer *mbuffer){
	for(int i = 0; i < mbuffer.elem_num; i++){
		free_message(&mbuffer->messages[i]);
	}
}


void free_message(Message_Buffer_Elem *elem){
	free(elem->message);
	free(elem);
}*/
