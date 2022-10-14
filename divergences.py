""" 
Implementation of the divergences. 
Free inspiration from :
    https://gitlab.com/i-flow/i-flow/, arXiv:2001.05486 (Tensorflow)
"""
import torch

class Divergence:
    """Divergence class conatiner.

    This class contains a list of f-divergences that
    can serve as loss functions in i-flow. f-divergences
    are a mathematical measure for how close two statistical
    distributions are.

    Note : mean() divides by tensor.numel() (= Number of elements in tensor != shape[0],
        except for 1D tensor)

    Attributes:
        alpha (float): attribute needed for (alpha, beta)-product divergence
            and Chernoff divergence
        beta (float): attribute needed for (alpha, beta)-product divergence

    """

    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.avail = [x for x in dir(self)
                            if ('__' not in x and 'alpha' not in x
                                and 'beta' not in x)]

    @staticmethod
    def MSE(true, test):
        """
        Implement classic MSE

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points

        Returns:
            tf.tensor(float): computed MSE divergence
        """
        return (true-test).pow(2).mean()


    @staticmethod
    def chi2(true, test):
        """
        Implement Neyman chi2 divergence.

        This function returns the Neyman chi2 divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points

        Returns:
            tf.tensor(float): computed Neyman chi2 divergence

        """
        return ((true-test).pow(2)/test).mean()

    @staticmethod
    def KL(true, test):
        """
        Implement Kullback-Leibler 

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points

        Returns:
            tf.tensor(float): computed Neyman chi2 divergence

        """
        return (true*torch.log(true/test)).mean()

    
