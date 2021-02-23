"""
A few functions used for converting to and from quaternions and for averaging
orientations based on the quaternion representation.

Michal Kalkowski 26/10/2018, edit 20/11/2019
"""
import numpy as np


def average_quats(quats):
    """
    Calculates an unwieghted average of quaternions.

    This is a Python version of Tolga Birdal's MATLAB algorithm for quaternion
    averaging based on:
        http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    The original MATLAB file can be found in:
    https://github.com/tolgabirdal/averaging_quaternions

    Based on:
    Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
    "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
    no. 4 (2007): 1193-1197.

    Parameters:
    ---
    quats: ndarray, n by 4 array with quaternions to average arranged in rows

    Returns:
    ---
    avgs: ndarray, average quaternion
    """
    # Handle the anti-podal configuration
    # quats[quats[:, 0] < 0, 0] *= -1
    accumulator = np.zeros([4, 4])
    for i in range(quats.shape[0]):
        this_quat = quats[i, :].reshape(-1, 1)
        # Rank-1 update
        accumulator += this_quat.dot(this_quat.T)
    # Scale
    accumulator *= 1/quats.shape[0]

    # Get the eigenvector corresponding to the largest eigenvalue
    eig_val, avg = np.linalg.eig(accumulator)
    avgs = avg[:, np.argmax(abs(eig_val))]
    return avgs


def test_average_quats(how_many=100):
    """
    Tests quaternion averaging.

    Parameters:
    ---
    how_many: int, the number of trials
    """
    total_m_err = 0
    total_n_err = 0
    perturb = .5
    for i in range(how_many):
        # Generate a random quat
        quat_init = np.random.rand(4, 1)
        # Normalise
        quat_init *= 1/np.linalg.norm(quat_init)
        quats = np.tile(quat_init, (1, 10))
        quats += (np.random.rand(quats.shape[0], quats.shape[1]) - 0.5)*perturb

        avg_quats = average_quats(quats.T)
        errM = abs(avg_quats - quat_init).sum()/4

        # Simple averagins
        avg_naive = quats.mean(axis=1)
        avg_naive *= 1/np.linalg.norm(avg_naive)
        errN = abs(avg_naive - quat_init).sum()/4

        total_m_err += errM
        total_n_err += errN

        print(avg_quats)
        print(avg_naive)
        print(errM)
        print(errN)

    print(total_m_err)
    print(total_n_err)


def euler2quat(phi_1, Phi, phi_2, P=1):
    """
    Converts Euler angles according to Bunge conventnion (ZXZ) to quaternions.
    Based on:
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
        http://iopscience.iop.org/article/10.1088/0965-0393/23/8/083501
        
    Parameters:
    ---
    phi_1: ndarray, the first Euler angle; defined between 0 and 2pi
    Phi: ndarray, the second Euler angle; defined between 0 and pi
    phi_2: ndarray, the third Euler angle; defined between 0 and pi
    P: int, P parameter reponsible for signs of angles, as defined in the
       above-mentioned paper. Note that Dream3D uses P=1 convention, whereas
       neper uses P=-1 convention.
    Returns:
    ---
    quaternion: ndarray, quaternion angles (w, x, y, z)
    """
    if (phi_1 < 0).any() or (phi_1 > 2*np.pi).any():
        print('phi_1 must be within 0 and 2pi')
        return None
    if (phi_2 < 0).any() or (phi_2 > 2*np.pi).any():
        print('phi_2 must be within 0 and 2pi')
        return None
    if (Phi < 0).any() or (Phi > np.pi).any():
        print('Phi must be within 0 and pi')
        return None
    q1 = np.cos(0.5*Phi)*np.cos(0.5*(phi_1 + phi_2))
    q2 = -P*np.sin(0.5*Phi)*np.cos(0.5*(phi_1 - phi_2))
    q3 = -P*np.sin(0.5*Phi)*np.sin(0.5*(phi_1 - phi_2))
    q4 = -P*np.cos(0.5*Phi)*np.sin(0.5*(phi_1 + phi_2))
    quaternion = np.column_stack([q1, q2, q3, q4])
    quaternion[quaternion[:, 0] < 0] *= -1
    return quaternion


def quat2euler(q, P=1):
    """
    Converts quaternions to Euler angles.
    Based on http://iopscience.iop.org/article/10.1088/0965-0393/23/8/083501

    Parameters:
    ---
    q: ndarray, quaternion to convert, order (w, x, y, z)
    P: int, P parameter reponsible for signs of angles, as defined in the
       above-mentioned paper. Note that Dream3D uses P=1 convention, whereas
       neper uses P=-1 convention.
    Returns:
    ---
    eulers: ndarray, Euler angles
    """
    q = q.reshape(-1, 4)
    q = q.T
    q03 = q[0]**2 + q[3]**2
    q12 = q[1]**2 + q[2]**2
    chi = (q03*q12)**0.5

    case_1 = (chi == 0) & (q12 == 0)
    case_2 = (chi == 0) & (q03 == 0)
    case_3 = chi != 0

    euler_angles = np.zeros([3, q.shape[-1]])
    euler_angles[0, case_1] = np.arctan2(
        -2*P*q[0][case_1]*q[3][case_1], q[0][case_1]**2
        - q[3][case_1]**2)
    euler_angles[1, case_1] = 0
    euler_angles[2, case_1] = 0

    euler_angles[0, case_2] = np.arctan2(
        2*q[1][case_2]*q[2][case_2],
        q[1][case_2]**2 - q[2][case_2]**2)
    euler_angles[1, case_2] = np.pi
    euler_angles[2, case_2] = 0

    qq = q[:, case_3]
    euler_angles[0, case_3] = np.arctan2(
        (qq[1]*qq[3] - P*qq[0]*qq[2])/chi[case_3],
        (-P*qq[0]*qq[1] - qq[2]*qq[3])/chi[case_3])
    euler_angles[1, case_3] = np.arctan2(2*chi[case_3],
                                         q03[case_3] - q12[case_3])

    euler_angles[2, case_3] = np.arctan2(
        (qq[1]*qq[3] + P*qq[0]*qq[2])/chi[case_3],
        (qq[2]*qq[3] - P*qq[0]*qq[1])/chi[case_3])

    euler_angles[euler_angles < 0] += 2*np.pi
    return euler_angles.T
