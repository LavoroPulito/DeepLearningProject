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
        self.embed_seen=nn.Embedding(num_embeddings, embedding_dim) #da capire quanti sono in totale i pokemon visti (cambia mano a mano che vado avanti nella partita, all'inizio 6)
        # da capire qual è la dimensione dell'embedding per ciascuna feature

        #Features continue
        self.embed_stats=nn.Linear() 
        self.embed_stats_change=nn.Linear()
        self.embed_status=nn.Linear()
        self.embed_hp_ratio=nn.Linear()
        #da capire le dimensioni di input e output di ognuno dei comandi sopra


        #Campo
        self.embed_current_weather=nn.Embedding(num_embeddings,embedding_dim)
        self.embed_speed_modifier=nn.Embedding(num_embeddings,embedding_dim)

        #proiezione
        self.state_proj=nn.Linear(in_features=350, out_features=d_model) # 350 è la dimensione (totale) del token di stato


        #EMBEDDING DELL'AZIONE a_t
        self.embed_src=nn.Embedding(num_embeddings=2, embedding_dim)
        self.embed_dest=nn.Embedding(num_embeddings,embedding_dim)
        self.embed_mega=nn.Embedding(num_embeddings=2, embedding_dim)
        self.embed_slot_action=nn.Embedding(num_embeddings=6, embedding_dim)


        self.action_proj=nn.Linear(in_features, out_features=d_model)  #la dimensione di input è il doppio della somma delle dim di embedding (2 mosse)

        #EMBEDDING RICOMPENSA 
        self.embed_reward=nn.Embedding(num_embeddings=2,embedding_dim=d_model)


        #EMBEDDING DEL TURNO t
        self.embed_turn=nn.Embedding(num_embeddings=40, embedding_dim=d_model) #nummax turni in una partita=40

        


    def forward(self, state, battlefield, action, reward, turn):
        #i nostri token devono essere riorganizzati in tensori con dimensione (batch_size, turn, 12, ...)
        #batch_size=numero di partite, turn=numero di turni in una partita
        #12=numero di pokemon in uno stato

        #consideriamo separatamente in due tensori diversi il token di stato e il token del campo,
        #facciamo l'embedding separatamente e poi li concateniamo alla fine

        #l'azione è riorganizzata in un tensore di dimensioni (batch_size,2,...)

        batch_size=state.size(0)

        #EMBEDDING STATO s_t
        #Features discrete
        id_emb=self.embed_id(state['id'])
        type_emb=self.embed_type(state['type'])
        ability_emb=self.embed_ability(state['ability'])
        item_emb=self.embed_item(state['item'])
        slot_emb=self.embed_slot(state['slot'])
        seen_emb=self.embed_seen(state['seen'])

        #Features continue
        stats_emb=self.embed_stats(state['stats'])
        stats_change_emb=self.embed_stats_change(state['stats_change'])
        status_emb=self.embed_status(state['status'])
        hp_ratio_emb=self.embed_hp_ratio(state['hp_ratio'])


        #Concatenazione features di un singolo pokemon sull'ultima dimensione
        pokemon_emb=torch.cat([id_emb,type_emb,ability_emb,item_emb,slot_emb,seen_emb,stats_emb,stats_change_emb,status_emb,hp_ratio_emb], dim=-1)
        pokemon_flat=pokemon_emb.view(pokemon_emb.size(0),pokemon_emb.size(1),-1) #stiamo rendendo il tensore una lista piatta per ogni turno

        #EMBEDDING CAMPO
        current_weather_emb=self.embed_current_weather(battlefield['current_weather'])
        speed_modifier_emb=self.embed_speed_modifier(battlefield['speed_modifier'])

        #Concatenazione degli embedding di stato e campo
        full_state=torch.cat([pokemon_flat,current_weather_emb,speed_modifier_emb], dim=-1)
        #va proiettato in uno spazio di dimensione d_model
        state_emb=self.state_proj(full_state)

        #EMBEDDING AZIONE a_t e concatenazione
        src_emb=self.embed_src(action['src'])
        dest_emb=self.embed_dest(action['dest'])
        mega_emb=self.embed_mega(action['mega'])
        slot_action_emb=self.embed_slot_action(action['slot_action'])
        full_action=torch.cat([src_emb, dest_emb,mega_emb, slot_action_emb], dim=-1)
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
        
        return final_inputs





