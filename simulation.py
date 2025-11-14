import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def create_feature_matrix(n_mice=5, n_trials_per_mouse=40, learning_rate = 0.5):
    """
    Features represent behavioral and neural data from go-no-go olfactory discrimination task.

    PARAMETERS:
    - n_mice: the number of mice/observations in the study
    - n_trials_per_mouse: the number of trials per mouse simulated
    - learning_rate: A rate in [0,1] used to simulate how quickly the mice learn. 
                     Lower values mean slower progression between frequency levels.
                     Can enter -1 as an option to use default thresholds of 0.3 and 0.7.
                     If learning_rate is not -1 or in [0,1], uses a learning_rate of 0.5.
    
    BEHAVIORAL FEATURES (Binary/Categorical):
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
    
    NEURAL FEATURES (Continuous - basic properties):
    - theta_frequency: Dominant theta oscillation frequency (Hz) in olfactory bulb (5-9 Hz range).
      Theta rhythms are linked to sniffing and odor sampling. Remains relatively stable across 
      learning stages, reflecting the consistent respiratory rhythm during odor sampling.
      
    - beta_frequency: Dominant beta oscillation frequency (Hz) in olfactory bulb (15-30 Hz range).
      Beta rhythms are associated with odor discrimination. Increases with learning as neural 
      representations become more organized.
      
    - gamma_frequency: Dominant gamma oscillation frequency (Hz) in olfactory bulb (65-95 Hz range).
      Gamma rhythms are associated with active sensory processing and odor discrimination. 
      Increases with learning as task-relevant processing strengthens.
      
    - noise_level: Background noise level in neural recordings (arbitrary units, 0.5-2.0 range).
      Lower noise = clearer signals. Decreases with learning as representations become more precise
      and consistent, reflecting reduced dimensionality of neural activity patterns.
    """
    
    all_trials = []
    all_trial_types = []
    
    # Generate data for each mouse
    for mouse_id in range(1, n_mice + 1):

        # Each mouse has slightly different baseline neural properties (individual differences)
        mouse_theta_freq = np.random.uniform(8, 12)      # Base theta frequency for this mouse (stable)
        mouse_beta_freq = np.random.uniform(20, 26)     # Base beta frequency for this mouse
        mouse_gamma_freq = np.random.uniform(70, 80)    # Base gamma frequency for this mouse
        mouse_noise_level = np.random.uniform(0.8, 1.2) # Baseline noise level for this mouse
        
        for trial_num in range(1, n_trials_per_mouse + 1):
            
            # Determine learning stage based on trial progression
            # From study: Behavioral performance was termed naïve or proficient when their performance estimated in a 20-trial window was ≤65% for naïve and ≥80% for proficient.
            if learning_rate == -1:
                rate1 = 0.3
                rate2 = 0.7
            elif learning_rate <=1 and learning_rate >= 0:
                rate1 = 1 - learning_rate
                rate2 = 1 - learning_rate * learning_rate
            else:
                #use default values, given learning rate is invalid
                rate1 = 0.5
                rate2 = 0.75

            if trial_num <= n_trials_per_mouse * rate1:  # First set of trials
                learning_stage = 'Naive'
                # Low performance: around 60% accuracy
                probs = [0.30, 0.20, 0.30, 0.20]  # [Hit, Miss, CR, FA]
                
            elif trial_num <= n_trials_per_mouse * rate2:  # Middle set of trials
                learning_stage = 'Learning'
                # Intermediate performance: around 75% accuracy
                probs = [0.375, 0.125, 0.375, 0.125]
                
            else:  # Last set of trials
                learning_stage = 'Proficient'
                # High performance: around 90% accuracy
                probs = [0.45, 0.05, 0.45, 0.05]
            
            # # Generate trial outcome based on learning stage probabilities
            trial_type = np.random.choice(['Hit', 'Miss', 'CR', 'FA'], p=probs)
            is_s_plus = trial_type in ['Hit', 'Miss']
            
            # Neural features
            # 1. Theta Frequency (Hz)
            # Remains relatively stable - linked to respiratory/sniffing rhythm
            theta_freq = mouse_theta_freq + np.random.normal(0, 0.3)
            theta_freq = np.clip(theta_freq, 6, 14)  # Keep in theta range
            
            # 2. Beta Frequency (Hz)
            # Increases with cognitive engagement and learning
            if learning_stage == 'Naive':
                beta_freq = mouse_beta_freq + np.random.normal(0, 1.0)
            elif learning_stage == 'Learning':
                beta_freq = mouse_beta_freq + 1.5 + np.random.normal(0, 1.0)
            else:  # Proficient
                # Higher for correct responses
                if trial_type in ['Hit', 'CR']:
                    beta_freq = mouse_beta_freq + 3.5 + np.random.normal(0, 1.0)
                else:
                    beta_freq = mouse_beta_freq + 2.0 + np.random.normal(0, 1.0)
            
            beta_freq = np.clip(beta_freq, 15, 30)  # Keep in beta range
            
            # 3. Gamma Frequency (Hz)
            # Increases with learning and active sensory processing
            if learning_stage == 'Naive':
                gamma_freq = mouse_gamma_freq + np.random.normal(0, 2.0)
            elif learning_stage == 'Learning':
                gamma_freq = mouse_gamma_freq + 3.0 + np.random.normal(0, 2.0)
            else:  # Proficient
                # Higher for S+ (more engaged processing)
                gamma_freq = mouse_gamma_freq + (7.0 if is_s_plus else 5.0) + np.random.normal(0, 2.0)
            
            gamma_freq = np.clip(gamma_freq, 65, 95)  # Keep in gamma range
            
            # 4. Noise Level (arbitrary units)
            # Decreases with learning (clearer signals)
            if learning_stage == 'Naive':
                noise = mouse_noise_level + np.random.normal(0, 0.1)
            elif learning_stage == 'Learning':
                noise = mouse_noise_level - 0.2 + np.random.normal(0, 0.1)
            else:  # Proficient
                # Lower noise for S+ (clearer representation)
                noise = mouse_noise_level - (0.4 if is_s_plus else 0.3) + np.random.normal(0, 0.1)
            
            noise = np.clip(noise, 0.3, 2.0)
            
            # Create feature dictionary
            trial_features = {
                # Behavioral features
                'Mouse_ID': mouse_id,
                'Trial_Number': trial_num,
                'Learning_Stage': learning_stage,
                'OdorOn': 1,
                'Hit': 1 if trial_type == 'Hit' else 0,
                'Miss': 1 if trial_type == 'Miss' else 0,
                'CR': 1 if trial_type == 'CR' else 0,
                'FA': 1 if trial_type == 'FA' else 0,
                'S+': 1 if is_s_plus else 0,
                'S-': 0 if is_s_plus else 1,
                'Reinf': 1 if trial_type == 'Hit' else 0,
                
                # Neural features (continuous)
                'theta_frequency': theta_freq,
                'beta_frequency': beta_freq,
                'gamma_frequency': gamma_freq,
                'noise_level': noise,
            }
            
            all_trials.append(trial_features)
            all_trial_types.append(trial_type)
    
    df = pd.DataFrame(all_trials)
    
    return df, all_trial_types

# Generate the data
feature_matrix, trial_labels = create_feature_matrix(n_mice=5, n_trials_per_mouse=40)

print("Feature Matrix Shape:", feature_matrix.shape)  # (200, 15)
print("\nFirst 10 trials:")
print(feature_matrix.head(10))

print("\nFeature columns:")
print(feature_matrix.columns.tolist())

print("\nTrial type distribution:")
print(pd.Series(trial_labels).value_counts())

print("\nLearning Progression By Stage:")

for stage in ['Naive', 'Learning', 'Proficient']:
    stage_data = feature_matrix[feature_matrix['Learning_Stage'] == stage]
    n_trials = len(stage_data)
    accuracy = (stage_data['Hit'].sum() + stage_data['CR'].sum()) / n_trials if n_trials > 0 else 0
    
    s_plus_data = stage_data[stage_data['S+'] == 1]
    s_minus_data = stage_data[stage_data['S-'] == 1]
    
    print(f"\n{stage} Stage: {n_trials} trials, {accuracy:.1%} accuracy")
    print(f"  S+ trials: Theta={s_plus_data['theta_frequency'].mean():.2f} Hz, "
          f"Beta={s_plus_data['beta_frequency'].mean():.2f} Hz, "
          f"Gamma={s_plus_data['gamma_frequency'].mean():.2f} Hz, "
          f"Noise={s_plus_data['noise_level'].mean():.3f}")
    print(f"  S- trials: Theta={s_minus_data['theta_frequency'].mean():.2f} Hz, "
          f"Beta={s_minus_data['beta_frequency'].mean():.2f} Hz, "
          f"Gamma={s_minus_data['gamma_frequency'].mean():.2f} Hz, "
          f"Noise={s_minus_data['noise_level'].mean():.3f}")

print("\nNeural Feature Statistics:")
neural_features = ['theta_frequency', 'beta_frequency', 'gamma_frequency', 'noise_level']
print(feature_matrix[neural_features].describe())