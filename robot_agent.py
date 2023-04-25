import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
from robot_neural_network import FullyConnectedModel

class Agent:
    def __init__(self, model, discount_rate, learning_rate_actor, learning_rate_critic, epsilon,
                 hidden_layer_size, weight_decay, dropout_rate,
                 action_space_dimension, observation_space_dimension, num_wheels,
                ):
        
        # define the model parameters
        self.model = model
        self.discount_rate = discount_rate
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.epsilon = epsilon
        self.hidden_layer_size = hidden_layer_size
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        
        # dimension of action space (9 actions in our case)
        self.action_space = action_space_dimension
        # dimension of state space (9 actions in our case)
        self.state_space = observation_space_dimension
        # number of wheels
        self.num_wheels = num_wheels

        # define optimizers
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_critic)

        # initialise MAA2C
        if self.model == "MAA2C":
            self.initialise_MAA2C()
        # initialise A2C
        elif self.model == "A2C":
            self.initialise_A2C()
        # initialise Reinforce
        elif self.model == "Reinforce":
            self.initialise_reinforce()

    def initialise_reinforce(self):
        """initialise reinforce model
        """
        self.actions = []
        self.Reinforce = FullyConnectedModel(model = "Reinforce",
                                    hidden_layer_size = self.hidden_layer_size,
                                    weight_decay = self.weight_decay,
                                    dropout_rate = self.dropout_rate, 
                                    num_of_outputs = self.action_space)
        self.Reinforce.model_name = "Reinforce"
        self.Reinforce.checkpoint_path = os.path.join(self.Reinforce.checkpoint_dir,
                                                        self.Reinforce.model_name)
        self.Reinforce.compile(optimizer = self.optimizer_actor)

    def initialise_MAA2C(self):
        """initialise MAA2C model
        """
        self.actions = [0 for _ in range(self.num_wheels)]
        self.MAA2C_Actors = [0 for _ in range(self.num_wheels)]

        # initialise one actor for each wheel
        for i in range(self.num_wheels):
            self.MAA2C_Actors[i] = FullyConnectedModel(model = "MAA2C_Actor", 
                                            hidden_layer_size = self.hidden_layer_size, 
                                            weight_decay = self.weight_decay,
                                            dropout_rate = self.dropout_rate, 
                                            num_of_outputs = self.action_space)
            self.MAA2C_Actors[i].model_name = "MAA2C_Actor_" + str(i)
            self.MAA2C_Actors[i].checkpoint_path = os.path.join(self.MAA2C_Actors[i].checkpoint_dir, self.MAA2C_Actors[i].model_name)
            self.MAA2C_Actors[i].compile(optimizer = self.optimizer_actor)
        
        # initialise critic model
        self.MAA2C_Critic = FullyConnectedModel(model = "MAA2C_Critic",
                                    hidden_layer_size = self.hidden_layer_size, 
                                    weight_decay = self.weight_decay,
                                    dropout_rate = self.dropout_rate, 
                                    num_of_outputs = 1)
        self.MAA2C_Critic.model_name = "MAA2C_Critic"
        self.MAA2C_Critic.checkpoint_path = os.path.join(self.MAA2C_Critic.checkpoint_dir, self.MAA2C_Critic.model_name)
        self.MAA2C_Critic.compile(optimizer = self.optimizer_critic)
    
    def initialise_A2C(self):
        """initialise A2C Model
        """
        self.actions = [0 for _ in range(self.num_wheels)]
        self.A2C = FullyConnectedModel(model = "A2C",
                            hidden_layer_size = self.hidden_layer_size, 
                            weight_decay = self.weight_decay,
                            dropout_rate = self.dropout_rate, 
                            num_of_outputs = self.action_space)
        self.A2C.model_name = "A2C"
        self.A2C.checkpoint_path = os.path.join(self.A2C.checkpoint_dir,
                                                                self.A2C.model_name)
        self.A2C.compile(optimizer = self.optimizer_actor)
    
    # Update Weights for Tensorflow Models
    def update_weights(self, model_name, observations, reward, observations_prime, is_done, actor_observations=None):
        persistent = False
        action_probs_list = [0 for _ in range(self.num_wheels)]
        log_prob_list = action_probs_list.copy()
        actor_loss_list = action_probs_list.copy()

        if model_name == "MAA2C":
            persistent = True
            actor_state_list = action_probs_list.copy()
            probs_list = action_probs_list.copy()
            actor_gradients_list = action_probs_list.copy()
            critic_state_prime = tf.convert_to_tensor([observations_prime], dtype = tf.float32)

            for i in range(self.num_wheels):
                actor_state_list[i] = tf.convert_to_tensor([actor_observations], dtype = tf.float32)
        
        state = tf.convert_to_tensor([observations], dtype = tf.float32)
        reward = tf.convert_to_tensor(reward, dtype = tf.float32)

        if model_name == "A2C":
            state_prime = tf.convert_to_tensor([observations_prime], dtype = tf.float32)

        # calculate state and action probabilities
        with tf.GradientTape(persistent = persistent) as gt:
            if model_name == "MAA2C":
                model = self.MAA2C_Critic
                critic_state_value = model(state)
                critic_state_value_prime = model(critic_state_prime)
                critic_state_value = tf.squeeze(critic_state_value)
                critic_state_value_prime = tf.squeeze(critic_state_value_prime)

                # td error <- reward + discount rate * V(s') - V(s)
                td_error = reward + self.discount_rate * critic_state_value_prime * (1 - is_done) - critic_state_value
                critic_loss = td_error ** 2

            elif model_name == "A2C":
                model = self.A2C

                state_value, action_probs = self.A2C(state)
                state_value_prime, _ = self.A2C(state_prime)
                state_value = tf.squeeze(state_value)
                state_value_prime = tf.squeeze(state_value_prime)
                # td error <- reward + discount rate * V(s') - V(s)
                td_error = reward + self.discount_rate * state_value_prime * (1 - is_done) - state_value

            elif model_name == "Reinforce":
                model = self.Reinforce
                action_probs = self.Reinforce(state)
                # td error <- reward + discount rate * V(s')
                td_error = reward + self.discount_rate * (1 - is_done)

            # update probability list
            if model_name == "A2C" or model_name == "Reinforce":
                action_probs_list[0] = tfp.distributions.Categorical(probs = action_probs[0])
                log_prob_list[0] = action_probs_list[0].log_prob(self.actions[0])
                actor_loss_list[0] = -log_prob_list[0] * td_error
                critic_loss =  td_error ** 2

            # get probability list from each agent
            elif model_name == "MAA2C":
                for i in range(self.num_wheels):
                    probs_list[i] = self.MAA2C_Actors[i](actor_state_list[i])
                    action_probs_list[i] = tfp.distributions.Categorical(probs = probs_list[i])
                    log_prob_list[i] = action_probs_list[i].log_prob(self.actions[i])
                    actor_loss_list[i] = -log_prob_list[i] * td_error
            
            # get total loss
            if model_name == "A2C":
                total_loss = critic_loss + sum(actor_loss_list)
            elif model_name == "Reinforce":
                total_loss = sum(actor_loss_list)
            elif model_name == "MAA2C":
                total_loss = critic_loss + sum(actor_loss_list)
        
        if model_name == "A2C" or model_name == "Reinforce":
            # apply gradients to actor models
            gradients = gt.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        elif model_name == "MAA2C":
            # apply gradients to critic model for multi-agent A2C
            critic_gradients = gt.gradient(critic_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(critic_gradients, model.trainable_variables))

        # apply gradient to actor models for multi-agent A2C
        if model_name == "MAA2C":
            for i in range(self.num_wheels):
                actor_gradients_list[i] = gt.gradient(actor_loss_list[i], self.MAA2C_Actors[i].trainable_variables)
                self.MAA2C_Actors[i].optimizer.apply_gradients(zip(actor_gradients_list[i],
                                                                    self.MAA2C_Actors[i].trainable_variables))
        del gt

        return total_loss.numpy()
    
    def store_memory(self, state, action, reward, state_prime, is_done):
        self.memory.log(state, action, reward, state_prime, is_done)
    
    def select_actions(self, observations, mode):
        actions_list = []
        self.actions = []
        explore = True if np.random.random() < self.epsilon and mode != "test" else False

        if self.model == "MAA2C":
            if explore:
                for i in range(self.num_wheels):
                    action = np.random.choice(self.action_space)
                    self.actions.append(action)
                    actions_list.append(action)
            else:
                for i in range(self.num_wheels):
                    state = tf.convert_to_tensor([observations], dtype = tf.float32)
                    prob = self.MAA2C_Actors[i](state)
                    prob = prob.numpy()
                    prob = np.nan_to_num(prob)
                    action = np.argmax(prob)
                    self.actions.append(tf.convert_to_tensor([action], dtype = tf.float32))
                    actions_list.append(action)
        
        elif self.model == "A2C":
            if explore:
                action = np.random.choice(self.action_space)
                self.actions.append(action)
                actions_list.append(action)
            else:
                state = tf.convert_to_tensor([observations], dtype = tf.float32)
                _, prob = self.A2C(state)
                prob = prob[0].numpy()
                prob = np.nan_to_num(prob)
                action = np.argmax(prob)
                actions_list.append(action)
                self.actions.append(tf.convert_to_tensor([action], dtype = tf.float32))
        
        elif self.model == 'Reinforce':
            if explore:
                action = np.random.choice(self.action_space)
                self.actions.append(action)
                actions_list.append(action)
            else:
                state = tf.convert_to_tensor([observations], dtype = tf.float32)
                prob = self.Reinforce(state)
                prob = prob[0].numpy()
                prob = np.nan_to_num(prob)
                action = np.argmax(prob)
                self.actions.append(action)
                actions_list.append(action)
                self.actions[0] = tf.convert_to_tensor([action], dtype = tf.float32)

        return actions_list
    
    
    def save_models(self,path=''):
        """ Save models """

        print("Saving model weights")
        self.checkpoint_path_global=path+f"/{self.model}_"

        if self.model == "MAA2C":
            for i in range(self.num_wheels):
                self.MAA2C_Actors[i].checkpoint_path = self.checkpoint_path_global+f'wheel_{i}'
                self.MAA2C_Actors[i].save_weights(self.MAA2C_Actors[i].checkpoint_path)

            self.MAA2C_Critic.checkpoint_path = self.checkpoint_path_global+f'critic'
            self.MAA2C_Critic.save_weights(self.MAA2C_Critic.checkpoint_path)

        elif self.model == 'A2C':
            self.A2C.checkpoint_path = self.checkpoint_path_global+f'A2C'
            self.A2C.save_weights(self.A2C.checkpoint_path)

        if self.model == "Reinforce":
            
            self.Reinforce.checkpoint_path=self.checkpoint_path_global+f'policy'
            self.Reinforce.save_weights(self.Reinforce.checkpoint_path,overwrite=True,save_format="h5")
            print(self.Reinforce.checkpoint_path)


    def load_models(self,path=''):
        """ Loading models """
        print("Loading model weights")
        self.checkpoint_path_global=path+f"/{self.model}_"

        if self.model == "MAA2C":
            for i in range(self.num_wheels):
                self.MAA2C_Actors[i].checkpoint_path = self.checkpoint_path_global+f'wheel_{i}.h5'
                self.MAA2C_Actors[i].load_weights(self.MAA2C_Actors[i].checkpoint_path).expect_partial()

            self.MAA2C_Critic.checkpoint_path = self.checkpoint_path_global+f'critic.h5'
            self.MAA2C_Critic.load_weights(self.MAA2C_Critic.checkpoint_path).expect_partial()

        elif self.model == "A2C":
            self.A2C.checkpoint_path = self.checkpoint_path_global+f'A2C'
            self.A2C.load_weights(self.A2C.checkpoint_path).expect_partial()

        if self.model == "Reinforce":
            self.Reinforce.checkpoint_path=self.checkpoint_path_global+f'policy.h5'
            self.Reinforce.load_weights(self.Reinforce.checkpoint_path)

        