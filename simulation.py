import numpy as np
import pandas as pd

def create_feature_matrix(n_mice=5, n_trials_per_mouse=40):
    """
    Features represent behavioral trial metadata from go-no-go olfactory task:
    - Mouse_ID: Identifier for each mouse (1, 2, 3, ...)
    - Trial_Number: Sequential trial number for each mouse (1 to n_trials_per_mouse)
    - Learning_Stage: Categorical stage based on trial progression ('Naive', 'Learning', 'Proficient')
    - OdorOn: Indicates odor was presented (always 1 in this task)
    - Hit: Mouse correctly licked for rewarded odorant (S+) - correct response
    - Miss: Mouse failed to lick for rewarded odorant (S+) - error/omission
    - CR: Correct Rejection - mouse correctly withheld licking for unrewarded odorant (S-)
    - FA: False Alarm - mouse incorrectly licked for unrewarded odorant (S-) - error
    - S+: Trial presented the rewarded odorant (Hit or Miss trials)
    - S-: Trial presented the unrewarded odorant (CR or FA trials)
    - Reinf: Reinforcement (water reward) was delivered (only on Hit trials)
    """
    
    all_trials = []
    all_trial_types = []
    
    # Generate data for each mouse
    for mouse_id in range(1, n_mice + 1):
        
        for trial_num in range(1, n_trials_per_mouse + 1):
            
            # Determine learning stage based on trial progression
            # From study: Behavioral performance was termed naïve or proficient when their performance estimated in a 20-trial window was ≤65% for naïve and ≥80% for proficient.
            if trial_num <= n_trials_per_mouse * 0.3:  # First 30% of trials
                learning_stage = 'Naive'
                # Low performance: around 60% accuracy
                probs = [0.30, 0.20, 0.30, 0.20]  # [Hit, Miss, CR, FA]
                
            elif trial_num <= n_trials_per_mouse * 0.7:  # Middle 40% of trials
                learning_stage = 'Learning'
                # Intermediate performance: around 75% accuracy
                probs = [0.375, 0.125, 0.375, 0.125]
                
            else:  # Last 30% of trials
                learning_stage = 'Proficient'
                # High performance: around 90% accuracy
                probs = [0.45, 0.05, 0.45, 0.05]
            
            # Generate trial outcome based on learning stage probabilities
            trial_type = np.random.choice(['Hit', 'Miss', 'CR', 'FA'], p=probs)
            
            # Create feature dictionary for this trial
            trial_features = {
                'Mouse_ID': mouse_id,
                'Trial_Number': trial_num,
                'Learning_Stage': learning_stage,
                'OdorOn': 1,
                'Hit': 1 if trial_type == 'Hit' else 0,
                'Miss': 1 if trial_type == 'Miss' else 0,
                'CR': 1 if trial_type == 'CR' else 0,
                'FA': 1 if trial_type == 'FA' else 0,
                'S+': 1 if trial_type in ['Hit', 'Miss'] else 0,
                'S-': 1 if trial_type in ['CR', 'FA'] else 0,
                'Reinf': 1 if trial_type == 'Hit' else 0,
            }
            
            all_trials.append(trial_features)
            all_trial_types.append(trial_type)
    
    df = pd.DataFrame(all_trials)
    
    return df, all_trial_types


# Generate the data
feature_matrix, trial_labels = create_feature_matrix(n_mice=5, n_trials_per_mouse=40)

print("Feature Matrix Shape:", feature_matrix.shape)  # (200, 11)
print("\nFirst 10 trials:")
print(feature_matrix.head(10))

print("\nFeature columns:")
print(feature_matrix.columns.tolist())

print("\nTrial type distribution:")
print(pd.Series(trial_labels).value_counts())

print("\nFeature counts (check mutual exclusivity):")
print(feature_matrix[['Hit', 'Miss', 'CR', 'FA']].sum(axis=0))

print("\nLearning progression by stage:")
for stage in ['Naive', 'Learning', 'Proficient']:
    stage_data = feature_matrix[feature_matrix['Learning_Stage'] == stage]
    n_trials = len(stage_data)
    accuracy = (stage_data['Hit'].sum() + stage_data['CR'].sum()) / n_trials if n_trials > 0 else 0
    print(f"{stage:12s}: {n_trials:3d} trials, {accuracy:.1%} accuracy")