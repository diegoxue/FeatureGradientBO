from environment import *
from scipy.spatial.distance import pdist, squareform
from glob import glob

def cal_knn_entropy(samples, k = 3):
    dist_mat = squareform(pdist(samples, metric='euclidean'))
    dist_mat = np.sort(dist_mat)
    return np.log(dist_mat[:, k - 1]).mean()

# path = 'results\\SMA_CO_enum&gradOpt_no_diff\\Backup - gradOpt_budget_budget_10_q_1_exp_40_100\\5f6614c6-45-40-100.pkl'
# path = 'results\\SMA_CO_enum&gradOpt_no_diff\\enumeration\\96f851b6-45-40-100.pkl'
init_smpl_len = 10

def ke_traj(path: str):
    comps, _ = joblib.load(path)
    ke_buffer = []
    for i in range(init_smpl_len, len(comps)):
        ke_buffer.append(cal_knn_entropy(comps[:i]))

    return ke_buffer

    for exp_idx, ke_buffer in zip(np.arange(41, len(comps) + 1), ke_buffer):
        print(exp_idx, ke_buffer)

def ke_analyse(path_reg: str):
    paths = glob(path_reg)
    ke_trajs = [ke_traj(p) for p in paths]
    ke_trajs = np.array(ke_trajs)

    save_path = os.path.join(os.path.dirname(path_reg), '~state_entropy_res.txt')
    np.savetxt(
        save_path, 
        np.vstack((
            np.arange(len(ke_trajs.T)) + init_smpl_len + 1,
            ke_trajs.mean(axis = 0),
            ke_trajs.std(axis = 0),
        )).T, 
        delimiter = '\t'
    )

# ke_analyse('results\\SMA_CO_enum&gradOpt_no_diff\\Backup - gradOpt_budget_budget_10_q_1_exp_40_100\\*pkl')
ke_analyse('results\\results_hea_c4_enum_20240909\\*pkl')
ke_analyse('results\\results_hea_c4_grad_20240909\\*pkl')