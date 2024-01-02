### Project Overview
The goal of this project is to model mouse behavior using hidden Markov and generalized liear models.

I am using generalized linear model combined with hidden Markov model which is put into the package called ssm. The method of using GLM-HMM is described in the following [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8890994/),
and the ssm package could be found [here](https://github.com/lindermanlab/ssm/tree/master). Additionally, I am using the psytrack model described [here](https://www.cell.com/neuron/fulltext/S0896-6273(20)30963-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627320309636%3Fshowall%3Dtrue)
and the psytrack [package](https://github.com/nicholas-roy/psytrack/tree/master)

### Experiment Design
Mice go through training to recognize various grayscale images and learn to give responses by licking either to the right or left. After they learned the task, during the actual experiment, these images (referred to as stimuli) are presented to the mice randomly. The mice display varying behaviors during different intervals, including providing mostly correct responses, showing a bias toward one side (left or right), or showing minimal interaction.

GLM-HMM is a good choice of a model because mice express distinct types of behaviour that could be modeled as hidden Markov states. 
One downside of using GLM-HMM could be that it produses descrete states probability and doesn't describe transitions between the states very well. To capture state transitions, we are using psytrack, the model that outputs a set of continous evolving weights which might be able to capture state transitions more efficiently.
The inputs into both models are observed mouse choices and stimuli (images shown to mice). In addition to mouse choices, I am looking at mouse reaction times to see if they change in different states. If mouse reaction times add more information about each state, they should be included into the models as a separate parameter.

### Main project milestones

#### How many GLM-HMM states should we use?

In answering the initial question regarding the number of hidden Markov states for model fitting, I split the data into training and testing sets. Subsequently, I applied the GLM-HMM model to the training dataset with varying number of states from 10 to 50. The evaluation of the model's performance on the test dataset was based on the log-likelihood values.
I noticed that beyond 3-4 states, the improvement in log-likelihood was minimal which led me to conclusion that there was diminishing returns in model performance after this point. Therefore, a decision was made to proceed with using 3 hidden Markov states for further analysis.

#### Gain some GLM-HMM model intuition

To see how well the model recognises different states and where it breaks, I created a generative glm-hmm model with preset glm weights and hmm transition matrices. This step was important in order to evaluate how many trials and sessions we will need to perform with actual mice in order to produce reliable state data. I used the same values of weights and state probabilities as describer in the Ashwood et al, 2019 paper. I used the generative model to produce sets of different number of mouse choices. Then I created another glm-hmm model to fit each set of synthetic mouse choice data in order to test how well the model recognizes different states depending on the number of the input values. Additionally, I fitted fit each set of synthetic data points to the generative model for comparison. I plotted the true state of the generative model for each datapoint, in addition to the predicted states by the glm-hmm model and the generative glm-hmm model. Furthermore, I plotted the weights of the generative model and the glm-hmm model to test how well the weights of the generative model were recovered by the glm-hmm model with different number of mouse choices. I found that the number of data points lower than 500 doesn't produce reliable model weight recovery.

#### Fit Psytrack to synthetic mouse choice data

In order to compare performance of psytrack on mouse choice data, I generated mouse choices with the glm-hmm model and fit both psytrack and glm-hmm models to those data. Additionally, I generated mouse choices with psytrack and fitted both models to these data. Psytrack doesn't have a concept of states and is described by a progression of dynamic weights. To compare glm-hmm and psytrack, I calculated dynamic weights for each HMM states obtained after fitting glm-hmm by multiplying the state probability of each data points by it's bias and stimulus weights. I found that both models follow similar trends although they disagree in certain points. Currently I am working on metrics to compare both model performance on synthetic data.

#### Fit psytrack and glm-hmm to experimental data. 

After working primarily with syntetic data, I fitted both models to the experimental mouse choice data. The difference between our experiment and the one described in the Ashwood et al. paper is that in the papaper, researchers recorded two types of mouse choices: left choice and right choice. In our case, we collected three choices: left, right, and no response. The psytrack can't handle three choice data, and as a result, I filtered out no response choice from the experimental data. I fitted both psytrack and glm-hmm to the experimental data and found that glm-hmm performs better at describing different types of mouse behavior than psytrack. Different types of mouse behavior that were recognised by the two-choice glm-hmm model include: left-bias, right-bias, choosing both sides equally. 

Since glm-hmm can handle three types of mouse choices, I fitted it with three choices as well (left, right, no response). The three choice glm-hmm model performed the best in my case recognising the following types of mouse behavior (states): biased (either left or right), giving mostly correct answers (choosing left and right approximately equally), and no response.

#### Looking at mouse reaction times

There were several measured parameters in addition to recording mouse choice data. One of these parameters is mouse reaction times measured in milliseconds. We decided to look at the reaction times because if they could be an important charachteristic of the different states. I plotted reaction times along the states, and found from visually inspecting the graphs that in general the mouse slows down around transition points between the states. Additionally, I found that the right choice is generally slower than the left choice and the wrong choice is slower than the correct choice. 


