import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='UIAdaptation-v0',
    entry_point='ui_adapt.envs:UIAdaptationEnv',
    nondeterministic = True,
)
