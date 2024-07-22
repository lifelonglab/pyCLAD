### What is pyCLAD?

The pyCLAD library contains classes providing tools and methods to experiment with continual anomaly detection problems.

The library is meant to be as easy-to-use as possible, coupled with extensive documentation of all functions and examples of how to effectively use them.

The main goal of pyCLAD is to simplify cumbersome tasks related to the design of experimental pipelines for continual anomaly detection. 

pyCLAD provides implementations of common functionalities for continual anomaly detection off-the-shelf, facilitating data analysis and visualization, and minimizing the risk of error-prone tasks, and fostering replicability. 

It also provides a simple and convenient infrastructure for designing new strategies, models, and evaluation procedures, enabling researchers to avoid repetitive tasks and allowing them to focus on creative and scientific aspects, reducing the friction related to low-level implementation aspects. 

The core coding infrastructure is influenced by PyTorch, a very popular framework for deep learning as well as SkLearn, the leading machine learning and data analysis library in Python.

### What is continual anomaly detection?

**Continual Learning** puts emphasis on models that answering the need for machine learning models that continuously adapt to new challenges in dynamic environments while retaining past knowledge.

**Anomaly Detection** is the process of detecting deviations from the normal behavior of a process, and has a very wide range of applications including monitoring cyber-physical systems, human conditions, as well as network traffic.

**Continual anomaly detection** lies at the intersection of these two fields. Its strategies focus on specific goals which can be compared to other types of anomaly detection approaches. One possible categorization is the following:

- **Offline**: Models are trained once on background data and do not require updates (examples: post-incident analysis, breast cancer detection)  and  on updating the model as new data is observed, assuming only the most recent information is relevant. This approach is static in nature and does not provide adaptation. 

- **Online**: Models are updated as new data is observed, assuming that the most recent information is the most relevant. This approach is popular in real-world dynamic applications where adaptation is necessary, but makes models prone to forgetting past knowledge. 

- **Continual**: Models are updated to simultaneously consider *adaptation* to new conditions and *knowledge retention* of previously observed (and potentially recurring) conditions [1].
This behavior attempts to overcome limitations of both offline and online anomaly detection in complex scenarios.


### How do I install pyCLAD?

The recommended way of installing pyCLAD is with Conda:

#### Conda
conda install -c conda-forge pyclad

#### Pip
python -m pip install pyclad

All requirements will be installed using any of the above install methods.


### Citing pyCLAD

    @article{faber2024pyclad,
      title={pyCLAD: A Library for Continual Lifelong Anomaly Detection},
      author={Faber, Kamil and Corizzo, Roberto and Sniezynski, Bartlomiej and Japkowicz, Nathalie},
      journal={SoftwareX},
      volume={TBD},
      pages={TBD},
      year={TBD},
      publisher={Elsevier}
    }

### References

[1] Faber, K., Corizzo, R., Sniezynski, B., & Japkowicz, N. (2024). Lifelong Continual Learning for Anomaly Detection: New Challenges, Perspectives, and Insights. *IEEE Access*, 12, 41364-41380.
