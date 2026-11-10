"""Reinforcement-learning trainer for the state translator.

A Gymnasium environment (``TranslationEnv``) turns a BBOB run with optimizer
switches into an MDP; a custom PPO loop (``ppo``, policy
``policy.ActorCritic``) or TD3 loop (``td3``) trains the set-equivariant
``TranslatorPair`` as the actor to maximize true downstream optimization
improvement.
"""
