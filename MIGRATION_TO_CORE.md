# Migration From TF-Addons To TensorFlow Core

### In-Progress & Previous Migrations:
https://github.com/tensorflow/addons/projects/2/

### Process 
1. Create an issue in TensorFlow Addons for a candidate that you think should be 
migrated. 
2. The SIG will evaluate the request and add it to the `Potential Candidates` section 
of our GitHub project.
3. If it's agreed that a migration makes sense, an RFC needs to be written to discuss 
the move with a larger community audience. 
    * Additions which subclass Keras APIs should submit their migration proposals to 
    [Keras Governance](https://github.com/keras-team/governance)
    * Other additions should submit their migration proposals to 
    [TensorFlow Community](https://github.com/tensorflow/community)
4. If approved, a pull request must move the addition along with proper tests.
5. After merging, the addition will be replaced with an alias to the core function 
if possible.
6. If an alias is not possible (e.g. large parameter changes), then a deprecation 
warning will be added and eventual removed after 2 releases.


### Criteria for Migration
* The addition is widely used throughout the community
* The addition is unlikely to have API changes as time progresses
* The addition is well written / tested

