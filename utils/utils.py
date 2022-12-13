import numpy as np
import torch
import utils
import math


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False

def tile(x, n):
    if not is_positive_int(n):
        raise TypeError('Argument \'n\' must be a positive integer.')
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not utils.is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not utils.is_positive_int(num_dims):
        raise TypeError('Number of leading dims must be a positive integer.')
    if num_dims > x.dim():
        raise ValueError('Number of leading dims can\'t be greater than total number of dims.')
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not utils.is_positive_int(num_reps):
        raise TypeError('Number of repetitions must be a positive integer.')
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.
    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.
    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights,
        num_samples=num_samples,
        replacement=False
    )
    mask[indices] += 1
    return mask

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1-1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that
        sigmoid(temperature * max_value) = bound.
    If temperature is greater than 1, returns 1.
    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(- (1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


# Create a vector orthogonal to a passed one.
def ortho_vector(n: np.ndarray):
    p = np.array([0, 0, 0], float)
    if math.fabs(n[2]) > 0.0:
        k = math.sqrt(n[1] * n[1] + n[2] * n[2])
        p[0] = 0
        p[1] = -n[2] / k
        p[2] = n[1] / k
    else:
        k = math.sqrt(n[0] * n[0] + n[1] * n[1])
        p[0] = n[1] / k
        p[1] = -n[0] / k
        p[2] = 0

    return p

def ortho_vector_vectorized(n: np.ndarray):
    p = np.tile(np.array([0, 0, 0], float), (n.shape[0], 1))
    if math.fabs(n[0, 2]) > 0.0:
        k = np.sqrt(n[:, 1] * n[:, 1] + n[:, 2] * n[:, 2])
        p[:, 0] = 0
        p[:, 1] = -n[:, 2] / k
        p[:, 2] = n[:, 1] / k
    else:
        k = np.sqrt(n[:, 0] * n[:, 0] + n[:, 1] * n[:, 1])
        p[:, 0] = n[:, 1] / k
        p[:, 1] = -n[:, 0] / k
        p[:, 2] = 0
    return p

def cartesian_to_spherical(vec: np.ndarray):
    phi = math.atan2(vec[2], vec[0])
    if phi < 0:
        phi += 2 * math.pi
    theta = math.acos(vec[1])
    return np.array([phi, theta])

def cartesian_to_spherical_vectorized(vec: np.ndarray):
    phi = np.arctan2(vec[:, 2], vec[:, 0])
    phi = np.array(list(map(lambda t: t + 2 * np.pi if t < 0 else t, phi)))
    theta = np.arccos(vec[:, 1])
    return np.vstack([phi, theta]).T

# Maps sample in the unit square onto a hemisphere,
# defined by a normal vector with a cosine-weighted distribution
# with power e.
def cos_weight(s: np.ndarray, n: np.ndarray, e=1.):
    #Construct basis
    u = ortho_vector(n)
    v = np.cross(u, n)
    u = np.cross(n, v)

    # Calculate 2D sample
    r1 = s[0]
    r2 = s[1]

    # Transform to spherical coordinates
    sin_psi = math.sin(2. * np.pi*r1)
    cos_psi = math.cos(2. * np.pi*r1)
    cos_theta = math.pow(1. - r2, 1. / (e + 1.))
    sin_theta = math.sqrt(1. - cos_theta * cos_theta)

    vec = u * sin_theta * cos_psi + v * sin_theta * sin_psi + n * cos_theta
    norm_vec = vec / np.linalg.norm(vec)
    pdf = norm_vec[1] / np.pi
    light = cartesian_to_spherical(norm_vec)

    # Return the result
    return light, pdf

def cos_weight_vectorized(s: np.ndarray, n: np.ndarray, e=1.):
    #Construct basis
    u = ortho_vector_vectorized(n)
    v = np.cross(u, n)
    u = np.cross(n, v)

    # Calculate 2D sample
    r1 = s[:, 0]
    r2 = s[:, 1]

    # Transform to spherical coordinates
    sin_psi = np.sin(2. * np.pi*r1)
    cos_psi = np.cos(2. * np.pi*r1)
    cos_theta = np.power(1. - r2, 1. / (e + 1.))
    sin_theta = np.sqrt(1. - cos_theta * cos_theta)

    vec = u * sin_theta[:, None] * cos_psi[:, None] + \
          v * sin_theta[:, None] * sin_psi[:, None] + \
          n * cos_theta[:, None]
    norm_vec = vec / np.linalg.norm(vec, axis=1)[:, None]
    pdf = norm_vec[:, 1] / np.pi
    light = cartesian_to_spherical_vectorized(norm_vec)

    # Return the result
    return light, pdf

def uniform(eps: np.ndarray):
    eps[0] *= 2 * math.pi
    eps[1] = math.acos(eps[1])
    pdf = 1 / (2 * np.pi)
    return eps, pdf

def get_pdf_by_samples_uniform(lights: np.ndarray):
    y = np.ones(lights.shape[0])
    return torch.from_numpy(y / (2 * np.pi))

def spherical_to_cartesian(lights: np.ndarray):
    light_sample_dir_phi = lights[:, 0]
    light_sample_dir_theta = lights[:, 1]
    y = np.cos(light_sample_dir_theta)
    return y

def get_pdf_by_samples(lights: np.ndarray):
    y = spherical_to_cartesian(lights)
    return torch.from_numpy(y / np.pi)

# Getting samples similar to HIBRID
def get_test_samples(points: np.ndarray):

    norm_pts = points[:, 3:]
    lights = np.zeros([norm_pts.shape[0], 2], dtype=np.float32)
    pdfs = np.zeros(norm_pts.shape[0], dtype=np.float32)

    for i in range(norm_pts.shape[0]):
        s = np.random.uniform(0., 1., 2)
        norm = np.array([0, 1, 0])
        #[light, pdf] = cos_weight(s, norm)
        [light, pdf] = uniform(s)
        lights[i] = light
        pdfs[i] = pdf
    return [torch.from_numpy(lights).to('cpu'), torch.from_numpy(pdfs).to('cpu')]


def get_test_samples_vectorized(points: np.ndarray):
    points = points
    s = np.random.uniform(0., 1., (points.shape[0], 2))
    norm = np.tile(np.array([0, 1, 0]), (points.shape[0], 1))
    [lights, pdfs] = cos_weight_vectorized(s, norm)

    return [torch.from_numpy(lights).to('cpu'), torch.from_numpy(pdfs).to('cpu')]
