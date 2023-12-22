The psytrack_glmhmm_weight_compar.py script in this folder is used to generate simulation data 
with either glmhmm or psytrack models, fit both psytrack or hmmglm to the synthetic data and plot for
comparison.

Graphs in the graph folder:

1. glmhmm_psytrack_large_img_nobias_session_10000_.png
Contains a very lagre figure that was partially used for a presentation. The subplots of this 
figure contain: glm-hmm states of the fitted synthetic data
                synthetic mouse choices
                dynamic weights for psytrack and glm-hmm fitted to the data
the first subplot is empty because initally it contained glm-hmm stimulus and bias weights, but
eventually, I decided that they don't add much information

2. psdata_fixed_number_psy_session_10000_.png
To create this plot, I generated mouse choices using psytrack and fitted both model to those choices
To generate mouse choices for psytrack, I wrote a different module that could be found in this folder called
sim_psy_data.py

The graph represents psytrack dynamic weights for fitted to the syntetic data psytrack model, glm-hmm
model, and the true psytrack weights

3.  psdata_fixed_number_psy_session_60000_.png

Is similar to the previous graaph, but it contains larger number of generated data points. 
