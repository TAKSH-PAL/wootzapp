import verifiers as vf
from dataset import build_dataset
from rubric import build_rubric

def load_environment(**kwargs) -> vf.Environment:
    """
    Loads the Mobile UI Agent RL Environment with 30 tasks split into 
    20 training and 10 evaluation rollouts and a weighted reward rubric.
    """
    dataset = build_dataset(split="train")
    eval_dataset = build_dataset(split="eval")
    rubric = build_rubric()
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric
    )
