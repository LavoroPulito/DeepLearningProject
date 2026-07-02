#Embedding stato, azione, ricompensa e turno

import torch
import torch.nn as nn


class DecisionalTransformer(nn.Module):
    def __init__(self,d_model=256):
        super().__init__()
        self.d_model=d_model #dimensione complessiva dello spazio degli embeddings (=256?)
        #EMBEDDING DELLO STATO s_t
        #Da ripetere per ognuno dei 12 pokemon
        


        #Features discrete
        self.embed_id=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale gli id  
        self.embed_type=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale i tipi
        self.embed_ability=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale le abilità
        self.embed_item=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale gli strumenti
        self.embed_slot=nn.Embedding(num_embeddings, embedding_dim) #a capire quanti sono in totale gli slot (4?)
        # da capire qual è la dimensione dell'embedding per ciascuna feature

        #Features continue
        self.embed_stats=nn.Linear() 
        self.embed_stats_change=nn.Linear()
        self.embed_status=nn.Linear()
        self.embed_hp_ratio=nn.Linear()
        #da capire le dimensioni di input e output di ognuno dei comandi sopra

        #Mosse
        self.embed_id_move=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale gli id delle mosse
        self.embed_d_class=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale le classi di danno
        self.embed_t_class=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale le classi di tipo

        #Mosse continue
        self.embed_power=nn.Linear() #da capire le dimensioni di input e output
        self.embed_priority=nn.Linear() #da capire le dimensioni di input e output
        self.embed_accuracy=nn.Linear() #da capire le dimensioni di input e output

        #Campo
        self.embed_current_weather=nn.Embedding(num_embeddings=5,embedding_dim)
        self.embed_speed_modifier=nn.Linear()

        #proiezione
        self.state_proj=nn.Linear(in_features=550, out_features=d_model) # 350 è la dimensione (totale) del token di stato


        #EMBEDDING DELL'AZIONE a_t
        self.embed_player_user=nn.Embedding(num_embeddings=2, embedding_dim)
        self.embed_slot_user=nn.Embedding(num_embeddings,embedding_dim)
        self.embed_player_target=nn.Embedding(num_embeddings=2, embedding_dim)
        self.embed_slot_target=nn.Embedding(num_embeddings,embedding_dim)
        self.embed_mega=nn.Embedding(num_embeddings=2, embedding_dim)
        self.embed_move=nn.Embedding(num_embeddings=6, embedding_dim)


        self.action_proj=nn.Linear(in_features, out_features=d_model)  #la dimensione di input è il doppio della somma delle dim di embedding (2 mosse)

        #EMBEDDING RICOMPENSA 
        self.embed_reward=nn.Embedding(num_embeddings=2,embedding_dim=d_model)


        #EMBEDDING DEL TURNO t
        self.embed_turn=nn.Embedding(num_embeddings=40, embedding_dim=d_model) #nummax turni in una partita=40

        


    def forward(self, state, move, battlefield, action, reward, turn,attention_mask=None):
        #i nostri token devono essere riorganizzati in tensori con dimensione (batch_size, turn, 12, ...)
        #batch_size=numero di partite, turn=numero di turni in una partita
        #12=numero di pokemon in uno stato

        #move ha dimensione (batch_size,turn,12,4)

        #consideriamo separatamente in tre tensori diversi il token di stato, le mosse e il token del campo,
        #facciamo l'embedding separatamente e poi li concateniamo alla fine

        #l'azione è riorganizzata in un tensore di dimensioni (batch_size,2,...)

        batch_size=state['id'].size(0)

        #EMBEDDING STATO s_t
        #Features discrete
        id_emb=self.embed_id(state['id'])
        type_emb=self.embed_type(state['type'])
        ability_emb=self.embed_ability(state['ability'])
        item_emb=self.embed_item(state['item'])
        slot_emb=self.embed_slot(state['slot'])
        

        #Features continue
        stats_emb=self.embed_stats(state['stats'])
        stats_change_emb=self.embed_stats_change(state['stats_change'])
        status_emb=self.embed_status(state['status'])
        hp_ratio_emb=self.embed_hp_ratio(state['hp_ratio'])

        #Mosse
        #move ha dimensione (batch_size,turn,12,4)
        id_move_emb=self.embed_id_move(move['id_move'])
        d_class_emb=self.embed_d_class(move['d_class'])
        t_class_emb=self.embed_t_class(move['t_class'])
        power_emb=self.embed_power(move['power'])
        priority_emb=self.embed_priority(move['priority'])
        accuracy_emb=self.embed_accuracy(move['accuracy'])


        #Concatenazione features di un singolo pokemon sull'ultima dimensione
        pokemon_emb=torch.cat([id_emb,type_emb,ability_emb,item_emb,slot_emb,stats_emb,stats_change_emb,status_emb,hp_ratio_emb], dim=-1)
        pokemon_flat=pokemon_emb.view(pokemon_emb.size(0),pokemon_emb.size(1),-1) #stiamo rendendo il tensore una lista piatta per ogni turno

        #Concatenazione features delle mosse di un singolo pokemon sull'ultima dimensione
        move_emb=torch.cat([id_move_emb,d_class_emb,t_class_emb,power_emb,priority_emb,accuracy_emb], dim=-1)
        move_flat=move_emb.view(move_emb.size(0),move_emb.size(1),-1)

        #EMBEDDING CAMPO
        current_weather_emb=self.embed_current_weather(battlefield['current_weather'])
        speed_modifier_emb=self.embed_speed_modifier(battlefield['speed_modifier'])

        #Concatenazione degli embedding di stato e campo
        full_state=torch.cat([pokemon_flat,move_flat,current_weather_emb,speed_modifier_emb], dim=-1)
        #va proiettato in uno spazio di dimensione d_model
        state_emb=self.state_proj(full_state)

        #EMBEDDING AZIONE a_t e concatenazione
        player_user_emb=self.embed_player_user(action['player_user'])
        slot_user_emb=self.embed_slot_user(action['slot_user'])
        player_target_emb=self.embed_player_target(action['player_target'])
        slot_target_emb=self.embed_slot_target(action['slot_target'])
        mega_emb=self.embed_mega(action['mega'])
        slot_move_emb=self.embed_move(action['move'])
        full_action=torch.cat([player_user_emb, slot_user_emb, player_target_emb, slot_target_emb, mega_emb, slot_move_emb], dim=-1)
        action_flat=full_action.view(full_action.size(0),full_action.size(1),-1) #stiamo rendendo il tensore una lista piatta per ogni turno

        action_emb=self.action_proj(action_flat)


        #EMBEDDING RICOMPENSA
        reward_emb=self.embed_reward(reward)

        #EMBEDDING TURNO t
        turn_emb=self.embed_turn(turn)

        state_emb=state_emb+turn_emb
        action_emb=action_emb+turn_emb
        reward_emb=reward_emb+turn_emb

        final_inputs=torch.stack([reward_emb,state_emb,action_emb], dim=1).permute(0,2,1,3).reshape(batch_size,3*turn,self.d_model)

        #Attention_mask
        if attention_mask is not None:
            stacked_attention_mask=attention_mask.repeat_interleave(3, dim=1) #ripetiamo l'attention mask per reward, state e action
        else:
            stacked_attention_mask=None

        #La maschera serve per fare in modo che per ogni partita si consideri lo stesso numero di turni. Si aggiungono quindi dei turni fittizi
        #in partite più corte. La maschera assegna 1 ai turni reali e 0 ai turni fittizi
        #La maschera viene creata nel momento in cui si crea il batch di partite e viene passata al modello come input. Viene poi ripetuta per reward, state e action
        
        return final_inputs, stacked_attention_mask





