### Project Overview
The goal of this project is to model mouse behavior using hidden Markov and generalized liear models.

I am using generalized linear model combined with hidden Markov model which is put into the package called ssm. The method of using GLM-HMM is described in the following [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8890994/),
and the ssm package could be found [here](https://github.com/lindermanlab/ssm/tree/master). Additionally, I am using the psytrack model described [here](https://www.cell.com/neuron/fulltext/S0896-6273(20)30963-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627320309636%3Fshowall%3Dtrue)
and the psytrack [package](https://github.com/nicholas-roy/psytrack/tree/master)

### Experiment Design
The mice are trained on a task of recognising different grayscale images and respond by leaking either right or left. During the experiment, the mice are shown the images randomly(later reffered as stimulus) and they alternate between 
intervals of different types of behavior: giving mostly correct answers, being one side (left or right) biased, mostly not interacting. 
GLM-HMM is a good choice of a model because mice express distinct types of behaviour that could be modeled as hidden Markov states. 
One downside of using GLM-HMM could be that it produses descrete states probability and doesn't describe transitions between the states very well. To capture state transitions, we are using psytrack, the model that outputs a set of continous evolwing weights which might be able to capture state transitions more efficiently.
The inputs into both models are observed mouse choices and stimulii (images shown to mice). In addition to mouse choices, I am looking at mouse reaction times to see if they change with different states. If mouse reaction times add more information about each state, they should be included into the models as a separate parameter.

### Main projectmile stones

#### How many GLM-HMM states should we use?
First question I asked: how many hidden Markov states should be used for model fitting. 
To answer this question, I splitted data into training and testing data sets. I fitted the GLM-HMM model with the training dataset and 10-50 number of states, and compared it's performance on the test dataset. I accessed log-likelihood value to evaluate model's performance,
and determined that after the state number 3-4 the improvement in log-likelihood was negligible. I decided to move forvard with 3 HMM states.

#### Gain some GLM-HMM model intuition

To see how well the model recognises different states and where it breaks. 


